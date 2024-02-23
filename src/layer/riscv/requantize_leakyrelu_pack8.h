// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

static void requantize_leakyrelu_pack8_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& scale_in_data, const Mat& scale_out_data, const Mat& bias_data, float slope, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    int scale_in_data_size = scale_in_data.w;
    int scale_out_data_size = scale_out_data.w;
    int bias_data_size = bias_data.w;
    int vl = 8;

    // int8(relu(v * scale_in) * scale_out)
    // int8_relu(v * (scale_in * scale_out))

    // int8(relu(v * scale_in + bias) * scale_out)
    // int8_relu(v * (scale_in * scale_out) + (bias * scale_out))

    if (bias_data_size == 0)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int* intptr = bottom_blob.channel(q);
            signed char* ptr = top_blob.channel(q);

            vfloat32m2_t _scale_in0 = scale_in_data_size == 1 ? vfmv_v_f_f32m2(scale_in_data[0], vl) : vle32_v_f32m2((const float*)scale_in_data + q * 8, vl);
            vfloat32m2_t _scale_out0 = scale_out_data_size == 1 ? vfmv_v_f_f32m2(scale_out_data[0], vl) : vle32_v_f32m2((const float*)scale_out_data + q * 8, vl);

            vfloat32m2_t _scale0 = vfmul_vv_f32m2(_scale_in0, _scale_out0, vl);
            vfloat32m2_t _slope = vfmv_v_f_f32m2(slope, vl);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                vfloat32m2_t _v01 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                vfloat32m2_t _v23 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr + 8, vl), vl);
                vfloat32m2_t _v45 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr + 16, vl), vl);
                vfloat32m2_t _v67 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr + 24, vl), vl);

                _v01 = vfmul_vv_f32m2(_v01, _scale0, vl);
                _v23 = vfmul_vv_f32m2(_v23, _scale0, vl);
                _v45 = vfmul_vv_f32m2(_v45, _scale0, vl);
                _v67 = vfmul_vv_f32m2(_v67, _scale0, vl);

                *(int64_t*)ptr = float2int8leakyrelu(_v01, _slope);
                *(int64_t*)(ptr + 8) = float2int8leakyrelu(_v23, _slope);
                *(int64_t*)(ptr + 16) = float2int8leakyrelu(_v45, _slope);
                *(int64_t*)(ptr + 24) = float2int8leakyrelu(_v67, _slope);

                intptr += 32;
                ptr += 32;
            }
            for (; i + 1 < size; i += 2)
            {
                vfloat32m2_t _v01 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                vfloat32m2_t _v23 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr + 8, vl), vl);

                _v01 = vfmul_vv_f32m2(_v01, _scale0, vl);
                _v23 = vfmul_vv_f32m2(_v23, _scale0, vl);

                *(int64_t*)ptr = float2int8leakyrelu(_v01, _slope);
                *(int64_t*)(ptr + 8) = float2int8leakyrelu(_v23, _slope);

                intptr += 16;
                ptr += 16;
            }
            for (; i < size; i++)
            {
                vfloat32m2_t _v01 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);

                _v01 = vfmul_vv_f32m2(_v01, _scale0, vl);

                *(int64_t*)ptr = float2int8leakyrelu(_v01, _slope);

                intptr += 8;
                ptr += 8;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int* intptr = bottom_blob.channel(q);
            signed char* ptr = top_blob.channel(q);

            vfloat32m2_t _scale_in0 = scale_in_data_size == 1 ? vfmv_v_f_f32m2(scale_in_data[0], vl) : vle32_v_f32m2((const float*)scale_in_data + q * 8, vl);
            vfloat32m2_t _scale_out0 = scale_out_data_size == 1 ? vfmv_v_f_f32m2(scale_out_data[0], vl) : vle32_v_f32m2((const float*)scale_out_data + q * 8, vl);
            vfloat32m2_t _bias0 = bias_data_size == 1 ? vfmv_v_f_f32m2(bias_data[0], vl) : vle32_v_f32m2((const float*)bias_data + q * 8, vl);

            vfloat32m2_t _scale0 = vfmul_vv_f32m2(_scale_in0, _scale_out0, vl);
            _bias0 = vfmul_vv_f32m2(_bias0, _scale_out0, vl);

            vfloat32m2_t _slope = vfmv_v_f_f32m2(slope, vl);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                vfloat32m2_t _v01 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                vfloat32m2_t _v23 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr + 8, vl), vl);
                vfloat32m2_t _v45 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr + 16, vl), vl);
                vfloat32m2_t _v67 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr + 24, vl), vl);

                _v01 = vfmacc_vv_f32m2(_bias0, _v01, _scale0, vl);
                _v23 = vfmacc_vv_f32m2(_bias0, _v23, _scale0, vl);
                _v45 = vfmacc_vv_f32m2(_bias0, _v45, _scale0, vl);
                _v67 = vfmacc_vv_f32m2(_bias0, _v67, _scale0, vl);

                *(int64_t*)ptr = float2int8leakyrelu(_v01, _slope);
                *(int64_t*)(ptr + 8) = float2int8leakyrelu(_v23, _slope);
                *(int64_t*)(ptr + 16) = float2int8leakyrelu(_v45, _slope);
                *(int64_t*)(ptr + 24) = float2int8leakyrelu(_v67, _slope);

                intptr += 32;
                ptr += 32;
            }
            for (; i + 1 < size; i += 2)
            {
                vfloat32m2_t _v01 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                vfloat32m2_t _v23 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr + 8, vl), vl);

                _v01 = vfmacc_vv_f32m2(_bias0, _v01, _scale0, vl);
                _v23 = vfmacc_vv_f32m2(_bias0, _v23, _scale0, vl);

                *(int64_t*)ptr = float2int8leakyrelu(_v01, _slope);
                *(int64_t*)(ptr + 8) = float2int8leakyrelu(_v23, _slope);

                intptr += 16;
                ptr += 16;
            }
            for (; i < size; i++)
            {
                vfloat32m2_t _v01 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);

                _v01 = vfmacc_vv_f32m2(_bias0, _v01, _scale0, vl);

                *(int64_t*)ptr = float2int8leakyrelu(_v01, _slope);

                intptr += 8;
                ptr += 8;
            }
        }
    }
}
