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

static void requantize_leakyrelu_pack4_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& scale_in_data, const Mat& scale_out_data, const Mat& bias_data, float slope, const Option& opt)
{
    int vl = 4;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;
    int outc = top_blob.c;
    int out_elempack = top_blob.elempack;

    int scale_in_data_size = scale_in_data.w;
    int scale_out_data_size = scale_out_data.w;
    int bias_data_size = bias_data.w;

    // int8(relu(v * scale_in) * scale_out)
    // int8_relu(v * (scale_in * scale_out))

    // int8(relu(v * scale_in + bias) * scale_out)
    // int8_relu(v * (scale_in * scale_out) + (bias * scale_out))

    if (out_elempack == 8)
    {
        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const int* intptr0 = bottom_blob.channel(q * 2);
                const int* intptr1 = bottom_blob.channel(q * 2 + 1);
                signed char* ptr = top_blob.channel(q);

                vfloat32m1_t _scale_in0 = scale_in_data_size == 1 ? vfmv_v_f_f32m1(scale_in_data[0], vl) : vle32_v_f32m1((const float*)scale_in_data + q * 8, vl);
                vfloat32m1_t _scale_in1 = scale_in_data_size == 1 ? vfmv_v_f_f32m1(scale_in_data[0], vl) : vle32_v_f32m1((const float*)scale_in_data + q * 8 + 4, vl);
                vfloat32m1_t _scale_out0 = scale_out_data_size == 1 ? vfmv_v_f_f32m1(scale_out_data[0], vl) : vle32_v_f32m1((const float*)scale_out_data + q * 8, vl);
                vfloat32m1_t _scale_out1 = scale_out_data_size == 1 ? vfmv_v_f_f32m1(scale_out_data[0], vl) : vle32_v_f32m1((const float*)scale_out_data + q * 8 + 4, vl);

                vfloat32m1_t _scale0 = vfmul_vv_f32m1(_scale_in0, _scale_out0, vl);
                vfloat32m1_t _scale1 = vfmul_vv_f32m1(_scale_in1, _scale_out1, vl);
                vfloat32m1_t _slope = vfmv_v_f_f32m1(slope, vl);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    vfloat32m1_t _v00 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr0, vl), vl);
                    vfloat32m1_t _v01 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr0 + 4, vl), vl);
                    vfloat32m1_t _v02 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr0 + 8, vl), vl);
                    vfloat32m1_t _v03 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr0 + 12, vl), vl);
                    vfloat32m1_t _v10 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr1, vl), vl);
                    vfloat32m1_t _v11 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr1 + 4, vl), vl);
                    vfloat32m1_t _v12 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr1 + 8, vl), vl);
                    vfloat32m1_t _v13 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr1 + 12, vl), vl);
                    _v00 = vfmul_vv_f32m1(_v00, _scale0, vl);
                    _v01 = vfmul_vv_f32m1(_v01, _scale0, vl);
                    _v02 = vfmul_vv_f32m1(_v02, _scale0, vl);
                    _v03 = vfmul_vv_f32m1(_v03, _scale0, vl);
                    _v10 = vfmul_vv_f32m1(_v10, _scale1, vl);
                    _v11 = vfmul_vv_f32m1(_v11, _scale1, vl);
                    _v12 = vfmul_vv_f32m1(_v12, _scale1, vl);
                    _v13 = vfmul_vv_f32m1(_v13, _scale1, vl);
                    *(int64_t*)ptr = float2int8leakyrelu(_v00, _v10, _slope);
                    *(int64_t*)(ptr + 8) = float2int8leakyrelu(_v01, _v11, _slope);
                    *(int64_t*)(ptr + 16) = float2int8leakyrelu(_v02, _v12, _slope);
                    *(int64_t*)(ptr + 24) = float2int8leakyrelu(_v03, _v13, _slope);

                    intptr0 += 16;
                    intptr1 += 16;
                    ptr += 32;
                }
                for (; i < size; i++)
                {
                    vfloat32m1_t _v0 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr0, vl), vl);
                    vfloat32m1_t _v1 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr1, vl), vl);
                    _v0 = vfmul_vv_f32m1(_v0, _scale0, vl);
                    _v1 = vfmul_vv_f32m1(_v1, _scale1, vl);
                    *(int64_t*)ptr = float2int8leakyrelu(_v0, _v1, _slope);

                    intptr0 += 4;
                    intptr1 += 4;
                    ptr += 8;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const int* intptr0 = bottom_blob.channel(q * 2);
                const int* intptr1 = bottom_blob.channel(q * 2 + 1);
                signed char* ptr = top_blob.channel(q);

                vfloat32m1_t _scale_in0 = scale_in_data_size == 1 ? vfmv_v_f_f32m1(scale_in_data[0], vl) : vle32_v_f32m1((const float*)scale_in_data + q * 8, vl);
                vfloat32m1_t _scale_in1 = scale_in_data_size == 1 ? vfmv_v_f_f32m1(scale_in_data[0], vl) : vle32_v_f32m1((const float*)scale_in_data + q * 8 + 4, vl);
                vfloat32m1_t _scale_out0 = scale_out_data_size == 1 ? vfmv_v_f_f32m1(scale_out_data[0], vl) : vle32_v_f32m1((const float*)scale_out_data + q * 8, vl);
                vfloat32m1_t _scale_out1 = scale_out_data_size == 1 ? vfmv_v_f_f32m1(scale_out_data[0], vl) : vle32_v_f32m1((const float*)scale_out_data + q * 8 + 4, vl);
                vfloat32m1_t _bias0 = bias_data_size == 1 ? vfmv_v_f_f32m1(bias_data[0], vl) : vle32_v_f32m1((const float*)bias_data + q * 8, vl);
                vfloat32m1_t _bias1 = bias_data_size == 1 ? vfmv_v_f_f32m1(bias_data[0], vl) : vle32_v_f32m1((const float*)bias_data + q * 8 + 4, vl);

                vfloat32m1_t _scale0 = vfmul_vv_f32m1(_scale_in0, _scale_out0, vl);
                vfloat32m1_t _scale1 = vfmul_vv_f32m1(_scale_in1, _scale_out1, vl);
                vfloat32m1_t _slope = vfmv_v_f_f32m1(slope, vl);
                _bias0 = vfmul_vv_f32m1(_bias0, _scale_out0, vl);
                _bias1 = vfmul_vv_f32m1(_bias1, _scale_out1, vl);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    vfloat32m1_t _v00 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr0, vl), vl);
                    vfloat32m1_t _v01 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr0 + 4, vl), vl);
                    vfloat32m1_t _v02 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr0 + 8, vl), vl);
                    vfloat32m1_t _v03 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr0 + 12, vl), vl);
                    vfloat32m1_t _v10 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr1, vl), vl);
                    vfloat32m1_t _v11 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr1 + 4, vl), vl);
                    vfloat32m1_t _v12 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr1 + 8, vl), vl);
                    vfloat32m1_t _v13 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr1 + 12, vl), vl);

                    _v00 = vfmacc_vv_f32m1(_bias0, _v00, _scale0, vl);
                    _v01 = vfmacc_vv_f32m1(_bias0, _v01, _scale0, vl);
                    _v02 = vfmacc_vv_f32m1(_bias0, _v02, _scale0, vl);
                    _v03 = vfmacc_vv_f32m1(_bias0, _v03, _scale0, vl);
                    _v10 = vfmacc_vv_f32m1(_bias1, _v10, _scale1, vl);
                    _v11 = vfmacc_vv_f32m1(_bias1, _v11, _scale1, vl);
                    _v12 = vfmacc_vv_f32m1(_bias1, _v12, _scale1, vl);
                    _v13 = vfmacc_vv_f32m1(_bias1, _v13, _scale1, vl);

                    *(int64_t*)ptr = float2int8leakyrelu(_v00, _v10, _slope);
                    *(int64_t*)(ptr + 8) = float2int8leakyrelu(_v01, _v11, _slope);
                    *(int64_t*)(ptr + 16) = float2int8leakyrelu(_v02, _v12, _slope);
                    *(int64_t*)(ptr + 24) = float2int8leakyrelu(_v03, _v13, _slope);

                    intptr0 += 16;
                    intptr1 += 16;
                    ptr += 32;
                }
                for (; i + 1 < size; i += 2)
                {
                    vfloat32m1_t _v00 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr0, vl), vl);
                    vfloat32m1_t _v01 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr0 + 4, vl), vl);
                    vfloat32m1_t _v10 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr1, vl), vl);
                    vfloat32m1_t _v11 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr1 + 4, vl), vl);

                    _v00 = vfmacc_vv_f32m1(_bias0, _v00, _scale0, vl);
                    _v01 = vfmacc_vv_f32m1(_bias0, _v01, _scale0, vl);
                    _v10 = vfmacc_vv_f32m1(_bias1, _v10, _scale1, vl);
                    _v11 = vfmacc_vv_f32m1(_bias1, _v11, _scale1, vl);

                    *(int64_t*)ptr = float2int8leakyrelu(_v00, _v10, _slope);
                    *(int64_t*)(ptr + 8) = float2int8leakyrelu(_v01, _v11, _slope);

                    intptr0 += 8;
                    intptr1 += 8;
                    ptr += 16;
                }
                for (; i < size; i++)
                {
                    vfloat32m1_t _v0 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr0, vl), vl);
                    vfloat32m1_t _v1 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr1, vl), vl);

                    _v0 = vfmacc_vv_f32m1(_bias0, _v0, _scale0, vl);
                    _v1 = vfmacc_vv_f32m1(_bias1, _v1, _scale1, vl);

                    *(int64_t*)ptr = float2int8leakyrelu(_v0, _v1, _slope);

                    intptr0 += 4;
                    intptr1 += 4;
                    ptr += 8;
                }
            }
        }
    }
    if (out_elempack == 1)
    {
        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                signed char* ptr0 = top_blob.channel(q * 4);
                signed char* ptr1 = top_blob.channel(q * 4 + 1);
                signed char* ptr2 = top_blob.channel(q * 4 + 2);
                signed char* ptr3 = top_blob.channel(q * 4 + 3);

                vfloat32m1_t _scale_in = scale_in_data_size == 1 ? vfmv_v_f_f32m1(scale_in_data[0], vl) : vle32_v_f32m1((const float*)scale_in_data + q * 4, vl);
                vfloat32m1_t _scale_out = scale_out_data_size == 1 ? vfmv_v_f_f32m1(scale_out_data[0], vl) : vle32_v_f32m1((const float*)scale_out_data + q * 4, vl);
                vfloat32m1_t _slope = vfmv_v_f_f32m1(slope, vl);

                vfloat32m1_t _scale = vfmul_vv_f32m1(_scale_in, _scale_out, vl);

                int i = 0;
                for (; i < size; i++)
                {
                    vfloat32m1_t _v = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr, vl), vl);
                    _v = vfmul_vv_f32m1(_v, _scale, vl);

                    int res = float2int8leakyrelu(_v, _slope);
                    ptr0[0] = (res)&0xff;
                    ptr1[0] = (res >> 8) & 0xff;
                    ptr2[0] = (res >> 16) & 0xff;
                    ptr3[0] = (res >> 24) & 0xff;

                    intptr += 4;
                    ptr0 += 1;
                    ptr1 += 1;
                    ptr2 += 1;
                    ptr3 += 1;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                signed char* ptr0 = top_blob.channel(q * 4);
                signed char* ptr1 = top_blob.channel(q * 4 + 1);
                signed char* ptr2 = top_blob.channel(q * 4 + 2);
                signed char* ptr3 = top_blob.channel(q * 4 + 3);

                vfloat32m1_t _scale_in = scale_in_data_size == 1 ? vfmv_v_f_f32m1(scale_in_data[0], vl) : vle32_v_f32m1((const float*)scale_in_data + q * 4, vl);
                vfloat32m1_t _scale_out = scale_out_data_size == 1 ? vfmv_v_f_f32m1(scale_out_data[0], vl) : vle32_v_f32m1((const float*)scale_out_data + q * 4, vl);
                vfloat32m1_t _bias = bias_data_size == 1 ? vfmv_v_f_f32m1(bias_data[0], vl) : vle32_v_f32m1((const float*)bias_data + q * 4, vl);
                vfloat32m1_t _slope = vfmv_v_f_f32m1(slope, vl);

                vfloat32m1_t _scale = vfmul_vv_f32m1(_scale_in, _scale_out, vl);
                _bias = vfmul_vv_f32m1(_bias, _scale_out, vl);

                int i = 0;
                for (; i < size; i++)
                {
                    vfloat32m1_t _v = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr, vl), vl);
                    _v = vfmacc_vv_f32m1(_bias, _v, _scale, vl);
                    int res = float2int8leakyrelu(_v, _slope);
                    ptr0[0] = (res)&0xff;
                    ptr1[0] = (res >> 8) & 0xff;
                    ptr2[0] = (res >> 16) & 0xff;
                    ptr3[0] = (res >> 24) & 0xff;

                    intptr += 4;
                    ptr0 += 1;
                    ptr1 += 1;
                    ptr2 += 1;
                    ptr3 += 1;
                }
            }
        }
    }
}
