// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 Xinyu302. All rights reserved.
// Copyright (C) 2019 BUG1989. All rights reserved.
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "quantize_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

#include "cpu.h"

namespace ncnn {

Quantize_riscv::Quantize_riscv()
{
#if __riscv_vector
    support_packing = true;

#if __riscv_zfh
    support_fp16_storage = true;
#endif // __riscv_zfh
#endif // __riscv_vector
}

int Quantize_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();
#if __riscv_vector && __riscv_zfh
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif // __riscv_vector && __riscv_zfh

    int vl = vsetvlmax_e32m1();
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

#if __riscv_vector
    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int out_elempack = opt.use_packing_layout && w * elempack % 8 == 0 ? 8 : 1;
            int outw = w * elempack / out_elempack;

            top_blob.create(outw, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                const float scale = scale_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const float* ptr0 = (const float*)bottom_blob + i * 4;
                    signed char* outptr = (signed char*)top_blob + i * 4;

                    outptr[0] = float2int8(ptr0[0] * scale);
                    outptr[1] = float2int8(ptr0[1] * scale);
                    outptr[2] = float2int8(ptr0[2] * scale);
                    outptr[3] = float2int8(ptr0[3] * scale);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const float* ptr0 = (const float*)bottom_blob + i * 4;
                    signed char* outptr = (signed char*)top_blob + i * 4;

                    outptr[0] = float2int8(ptr0[0] * scale_data[i * 4]);
                    outptr[1] = float2int8(ptr0[1] * scale_data[i * 4 + 1]);
                    outptr[2] = float2int8(ptr0[2] * scale_data[i * 4 + 2]);
                    outptr[3] = float2int8(ptr0[3] * scale_data[i * 4 + 3]);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int out_elempack = opt.use_packing_layout && h * elempack % 8 == 0 ? 8 : 1;
            int outh = h * elempack / out_elempack;

            top_blob.create(w, outh, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (out_elempack == 8)
            {
                if (scale_data_size == 1)
                {
                    // vfloat32m1_t _scale = vfmv_v_f_f32m1(scale_data[0]);
                    // float32x4_t _scale = vdupq_n_f32(scale_data[0]);
                    float _scale = scale_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outh; i++)
                    {
                        const float* ptr0 = bottom_blob.row(i * 2);
                        const float* ptr1 = bottom_blob.row(i * 2 + 1);
                        signed char* outptr = top_blob.row<signed char>(i);

                        for (int j = 0; j < w; j++)
                        {
                            vfloat32m1_t _vlow = vle32_v_f32m1(ptr0, vl);
                            vfloat32m1_t _vhigh = vle32_v_f32m1(ptr1, vl);
                            _vlow = vfmul_vf_f32m1(_vlow, _scale, vl);
                            _vhigh = vfmul_vf_f32m1(_vhigh, _scale, vl);

                            int64_t _v = float2int8(_vlow, _vhigh);
                            *(int64_t*)outptr = _v;

                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outh; i++)
                    {
                        const float* ptr0 = bottom_blob.row(i * 2);
                        const float* ptr1 = bottom_blob.row(i * 2 + 1);
                        signed char* outptr = top_blob.row<signed char>(i);

                        vfloat32m1_t _scale0 = vle32_v_f32m1((const float*)scale_data + i * 8, vl);
                        vfloat32m1_t _scale1 = vle32_v_f32m1((const float*)scale_data + i * 8 + 4, vl);

                        for (int j = 0; j < w; j++)
                        {
                            vfloat32m1_t _vlow = vle32_v_f32m1(ptr0, vl);
                            vfloat32m1_t _vhigh = vle32_v_f32m1(ptr1, vl);
                            _vlow = vfmul_vv_f32m1(_vlow, _scale0, vl);
                            _vhigh = vfmul_vv_f32m1(_vhigh, _scale1, vl);
                            int64_t _v = float2int8(_vlow, _vhigh);
                            *(int64_t*)outptr = _v;
                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
            }
            if (out_elempack == 1)
            {
                if (scale_data_size == 1)
                {
                    const float scale = scale_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < h; i++)
                    {
                        const float* ptr0 = bottom_blob.row(i);
                        signed char* outptr0 = top_blob.row<signed char>(i * 4);
                        signed char* outptr1 = top_blob.row<signed char>(i * 4 + 1);
                        signed char* outptr2 = top_blob.row<signed char>(i * 4 + 2);
                        signed char* outptr3 = top_blob.row<signed char>(i * 4 + 3);

                        for (int j = 0; j < w; j++)
                        {
                            outptr0[0] = float2int8(ptr0[0] * scale);
                            outptr1[0] = float2int8(ptr0[1] * scale);
                            outptr2[0] = float2int8(ptr0[2] * scale);
                            outptr3[0] = float2int8(ptr0[3] * scale);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < h; i++)
                    {
                        const float* ptr0 = bottom_blob.row(i);
                        signed char* outptr0 = top_blob.row<signed char>(i * 4);
                        signed char* outptr1 = top_blob.row<signed char>(i * 4 + 1);
                        signed char* outptr2 = top_blob.row<signed char>(i * 4 + 2);
                        signed char* outptr3 = top_blob.row<signed char>(i * 4 + 3);

                        const float s0 = scale_data[i * 4];
                        const float s1 = scale_data[i * 4 + 1];
                        const float s2 = scale_data[i * 4 + 2];
                        const float s3 = scale_data[i * 4 + 3];

                        for (int j = 0; j < w; j++)
                        {
                            outptr0[0] = float2int8(ptr0[0] * s0);
                            outptr1[0] = float2int8(ptr0[1] * s1);
                            outptr2[0] = float2int8(ptr0[2] * s2);
                            outptr3[0] = float2int8(ptr0[3] * s3);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;
            int out_elempack = opt.use_packing_layout && channels * elempack % 8 == 0 ? 8 : 1;
            int outc = channels * elempack / out_elempack;

            top_blob.create(w, h, outc, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (out_elempack == 8)
            {
                if (scale_data_size == 1)
                {
                    float _scale = scale_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < outc; q++)
                    {
                        const float* ptr0 = bottom_blob.channel(q * 2);
                        const float* ptr1 = bottom_blob.channel(q * 2 + 1);
                        signed char* outptr = top_blob.channel(q);

                        int i = 0;
                        for (; i + 1 < size; i += 2)
                        {
                            vfloat32m1_t _v0 = vle32_v_f32m1(ptr0, vl);
                            vfloat32m1_t _v1 = vle32_v_f32m1(ptr0 + 4, vl);
                            vfloat32m1_t _v2 = vle32_v_f32m1(ptr1, vl);
                            vfloat32m1_t _v3 = vle32_v_f32m1(ptr1 + 4, vl);
                            _v0 = vfmul_vf_f32m1(_v0, _scale, vl);
                            _v1 = vfmul_vf_f32m1(_v1, _scale, vl);
                            _v2 = vfmul_vf_f32m1(_v2, _scale, vl);
                            _v3 = vfmul_vf_f32m1(_v3, _scale, vl);

                            vint8m1_t _v = float2int8(_v0, _v2, _v1, _v3);
                            vse8_v_i8m1(outptr, _v, 4 * vl);
                            ptr0 += 8;
                            ptr1 += 8;
                            outptr += 16;
                        }
                        for (; i < size; i++)
                        {
                            vfloat32m1_t _vlow = vle32_v_f32m1(ptr0, vl);
                            vfloat32m1_t _vhigh = vle32_v_f32m1(ptr1, vl);

                            _vlow = vfmul_vf_f32m1(_vlow, _scale, vl);
                            _vhigh = vfmul_vf_f32m1(_vhigh, _scale, vl);

                            int64_t _v = float2int8(_vlow, _vhigh);
                            *(int64_t*)outptr = _v;
                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < outc; q++)
                    {
                        const float* ptr0 = bottom_blob.channel(q * 2);
                        const float* ptr1 = bottom_blob.channel(q * 2 + 1);
                        signed char* outptr = top_blob.channel(q);

                        vfloat32m1_t _scale0 = vle32_v_f32m1((const float*)scale_data + q * 8, vl);
                        vfloat32m1_t _scale1 = vle32_v_f32m1((const float*)scale_data + q * 8 + 4, vl);

                        int i = 0;
                        for (; i < size; i++)
                        {
                            vfloat32m1_t _vlow = vle32_v_f32m1(ptr0, vl);
                            vfloat32m1_t _vhigh = vle32_v_f32m1(ptr1, vl);

                            _vlow = vfmul_vv_f32m1(_vlow, _scale0, vl);
                            _vhigh = vfmul_vv_f32m1(_vhigh, _scale1, vl);

                            int64_t _v = float2int8(_vlow, _vhigh);
                            *(int64_t*)outptr = _v;
                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
            }
            if (out_elempack == 1)
            {
                if (scale_data_size == 1)
                {
                    const float scale = scale_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr0 = bottom_blob.channel(q);
                        signed char* outptr0 = top_blob.channel(q * 4);
                        signed char* outptr1 = top_blob.channel(q * 4 + 1);
                        signed char* outptr2 = top_blob.channel(q * 4 + 2);
                        signed char* outptr3 = top_blob.channel(q * 4 + 3);

                        for (int i = 0; i < size; i++)
                        {
                            outptr0[0] = float2int8(ptr0[0] * scale);
                            outptr1[0] = float2int8(ptr0[1] * scale);
                            outptr2[0] = float2int8(ptr0[2] * scale);
                            outptr3[0] = float2int8(ptr0[3] * scale);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr0 = bottom_blob.channel(q);
                        signed char* outptr0 = top_blob.channel(q * 4);
                        signed char* outptr1 = top_blob.channel(q * 4 + 1);
                        signed char* outptr2 = top_blob.channel(q * 4 + 2);
                        signed char* outptr3 = top_blob.channel(q * 4 + 3);

                        const float s0 = scale_data[q * 4];
                        const float s1 = scale_data[q * 4 + 1];
                        const float s2 = scale_data[q * 4 + 2];
                        const float s3 = scale_data[q * 4 + 3];

                        for (int i = 0; i < size; i++)
                        {
                            outptr0[0] = float2int8(ptr0[0] * s0);
                            outptr1[0] = float2int8(ptr0[1] * s1);
                            outptr2[0] = float2int8(ptr0[2] * s2);
                            outptr3[0] = float2int8(ptr0[3] * s3);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
            }
        }

        return 0;
    }
#endif // __riscv_vector

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* ptr = bottom_blob;
        signed char* outptr = top_blob;

        if (scale_data_size == 1)
        {
            const float scale = scale_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                outptr[i] = float2int8(ptr[i] * scale);
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                outptr[i] = float2int8(ptr[i] * scale_data[i]);
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const float* ptr0 = bottom_blob.row(i);
            signed char* outptr0 = top_blob.row<signed char>(i);

            const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

            for (int j = 0; j < w; j++)
            {
                *outptr0++ = float2int8(*ptr0++ * scale);
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(w, h, channels, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            signed char* outptr = top_blob.channel(q);

            const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

            int i = 0;
#if __riscv_vector
            float _scale = scale;

            for (; i + 15 < size; i += 16)
            {
                vfloat32m1_t _v0 = vle32_v_f32m1(ptr, vl);
                vfloat32m1_t _v1 = vle32_v_f32m1(ptr + 4, vl);
                vfloat32m1_t _v2 = vle32_v_f32m1(ptr + 8, vl);
                vfloat32m1_t _v3 = vle32_v_f32m1(ptr + 12, vl);

                _v0 = vfmul_vf_f32m1(_v0, _scale, vl);
                _v1 = vfmul_vf_f32m1(_v1, _scale, vl);
                _v2 = vfmul_vf_f32m1(_v2, _scale, vl);
                _v3 = vfmul_vf_f32m1(_v3, _scale, vl);

                vint8m1_t _v = float2int8(_v0, _v1, _v2, _v3);
                vse8_v_i8m1(outptr, _v, 4 * vl);

                ptr += 16;
                outptr += 16;
            }
            for (; i + 7 < size; i += 8)
            {
                vfloat32m1_t _v0 = vle32_v_f32m1(ptr, vl);
                vfloat32m1_t _v1 = vle32_v_f32m1(ptr + 4, vl);

                _v0 = vfmul_vf_f32m1(_v0, _scale, vl);
                _v1 = vfmul_vf_f32m1(_v1, _scale, vl);

                int64_t _v = float2int8(_v0, _v1);
                *(int64_t*)outptr = _v;
                ptr += 8;
                outptr += 8;
            }
#endif // __riscv_vector
            for (; i < size; i++)
            {
                *outptr++ = float2int8(*ptr++ * scale);
            }
        }
    }

    return 0;
}

#if __riscv_vector && __riscv_zfh

int Quantize_riscv::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;
    int vl;
    if (elempack == 8)
    {
        vl = 8;
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int out_elempack = opt.use_packing_layout && w * elempack % 8 == 0 ? 8 : 1;
            int outw = w * elempack / out_elempack;

            top_blob.create(outw, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                const float scale = scale_data[0];
                vfloat32m2_t _scale = vfmv_v_f_f32m2(scale, vl);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const __fp16* ptr0 = (const __fp16*)bottom_blob + i * 8;
                    signed char* outptr = (signed char*)top_blob + i * 8;

                    vl = 8;
                    vfloat16m1_t _v0 = vle16_v_f16m1(ptr0, vl);
                    vfloat32m2_t _v = vfwcvt_f_f_v_f32m2(_v0, vl);
                    _v = vfmul_vv_f32m2(_v, _scale, vl);
                    *(int64_t*)outptr = float2int8(vget_v_f32m2_f32m1(_v, 0), vget_v_f32m2_f32m1(_v, 1));
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const __fp16* ptr0 = (const __fp16*)bottom_blob + i * 8;
                    signed char* outptr = (signed char*)top_blob + i * 8;

                    vl = 8;
                    vfloat16m1_t _v0 = vle16_v_f16m1(ptr0, vl);
                    vfloat32m2_t _v = vfwcvt_f_f_v_f32m2(_v0, vl);
                    vfloat32m2_t _scale = vle32_v_f32m2((const float*)scale_data + i * 8, vl);
                    _v = vfmul_vv_f32m2(_v, _scale, vl);
                    *(int64_t*)outptr = float2int8(vget_v_f32m2_f32m1(_v, 0), vget_v_f32m2_f32m1(_v, 1));
                }
            }
        }
        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int out_elempack = opt.use_packing_layout && h * elempack % 8 == 0 ? 8 : 1;
            int outh = h * elempack / out_elempack;

            top_blob.create(w, outh, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                const float scale = scale_data[0];
                vfloat32m2_t _scale = vfmv_v_f_f32m2(scale, vl);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
                    signed char* outptr0 = top_blob.row<signed char>(i);

                    for (int j = 0; j < w; j++)
                    {
                        vl = 8;
                        vfloat16m1_t _v0 = vle16_v_f16m1(ptr0, vl);
                        vfloat32m2_t _v = vfwcvt_f_f_v_f32m2(_v0, vl);
                        _v = vfmul_vv_f32m2(_v, _scale, vl);
                        *(int64_t*)outptr0 = float2int8(vget_v_f32m2_f32m1(_v, 0), vget_v_f32m2_f32m1(_v, 1));

                        ptr0 += 8;
                        outptr0 += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
                    signed char* outptr0 = top_blob.row<signed char>(i);

                    vfloat32m2_t _scale = vle32_v_f32m2((const float*)scale_data + i * 8, vl);
                    for (int j = 0; j < w; j++)
                    {
                        vl = 8;
                        vfloat16m1_t _v0 = vle16_v_f16m1(ptr0, vl);
                        vfloat32m2_t _v = vfwcvt_f_f_v_f32m2(_v0, vl);
                        _v = vfmul_vv_f32m2(_v, _scale, vl);
                        *(int64_t*)outptr0 = float2int8(vget_v_f32m2_f32m1(_v, 0), vget_v_f32m2_f32m1(_v, 1));

                        ptr0 += 8;
                        outptr0 += 8;
                    }
                }
            }
        }
        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;

            int out_elempack = opt.use_packing_layout && channels * elempack % 8 == 0 ? 8 : 1;
            int outc = channels * elempack / out_elempack;

            top_blob.create(w, h, outc, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                const float scale = scale_data[0];
                vfloat32m2_t _scale = vfmv_v_f_f32m2(scale, vl);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr0 = bottom_blob.channel(q);
                    signed char* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        vl = 8;
                        vfloat16m1_t _v0 = vle16_v_f16m1(ptr0, vl);
                        vfloat32m2_t _v = vfwcvt_f_f_v_f32m2(_v0, vl);
                        _v = vfmul_vv_f32m2(_v, _scale, vl);
                        *(int64_t*)outptr = float2int8(vget_v_f32m2_f32m1(_v, 0), vget_v_f32m2_f32m1(_v, 1));
                        ptr0 += 8;
                        outptr += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr0 = bottom_blob.channel(q);
                    signed char* outptr = top_blob.channel(q);

                    vfloat32m2_t _scale = vle32_v_f32m2((const float*)scale_data + q * 8, vl);

                    for (int i = 0; i < size; i++)
                    {
                        vl = 8;
                        vfloat16m1_t _v0 = vle16_v_f16m1(ptr0, vl);
                        vfloat32m2_t _v = vfwcvt_f_f_v_f32m2(_v0, vl);
                        _v = vfmul_vv_f32m2(_v, _scale, vl);
                        *(int64_t*)outptr = float2int8(vget_v_f32m2_f32m1(_v, 0), vget_v_f32m2_f32m1(_v, 1));
                        ptr0 += 8;
                        outptr += 8;
                    }
                }
            }
        }
        return 0;
    }

    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int out_elempack = opt.use_packing_layout && w * elempack % 8 == 0 ? 8 : 1;
            int outw = w * elempack / out_elempack;

            top_blob.create(outw, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                const float scale = scale_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const __fp16* ptr0 = (const __fp16*)bottom_blob + i * 4;
                    signed char* outptr = (signed char*)top_blob + i * 4;

                    outptr[0] = float2int8((float)ptr0[0] * scale);
                    outptr[1] = float2int8((float)ptr0[1] * scale);
                    outptr[2] = float2int8((float)ptr0[2] * scale);
                    outptr[3] = float2int8((float)ptr0[3] * scale);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const __fp16* ptr0 = (const __fp16*)bottom_blob + i * 4;
                    signed char* outptr = (signed char*)top_blob + i * 4;

                    outptr[0] = float2int8((float)ptr0[0] * scale_data[i * 4]);
                    outptr[1] = float2int8((float)ptr0[1] * scale_data[i * 4 + 1]);
                    outptr[2] = float2int8((float)ptr0[2] * scale_data[i * 4 + 2]);
                    outptr[3] = float2int8((float)ptr0[3] * scale_data[i * 4 + 3]);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int out_elempack = opt.use_packing_layout && h * elempack % 8 == 0 ? 8 : 1;
            int outh = h * elempack / out_elempack;

            top_blob.create(w, outh, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (out_elempack == 8)
            {
                if (scale_data_size == 1)
                {
                    float _scale = scale_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outh; i++)
                    {
                        const __fp16* ptr0 = bottom_blob.row<const __fp16>(i * 2);
                        const __fp16* ptr1 = bottom_blob.row<const __fp16>(i * 2 + 1);
                        signed char* outptr = top_blob.row<signed char>(i);
                        vl = 4;

                        for (int j = 0; j < w; j++)
                        {
                            vfloat32m1_t _vlow = vget_v_f32m2_f32m1(vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr0, vl), vl), 0);
                            vfloat32m1_t _vhigh = vget_v_f32m2_f32m1(vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr1, vl), vl), 0);
                            _vlow = vfmul_vf_f32m1(_vlow, _scale, vl);
                            _vhigh = vfmul_vf_f32m1(_vhigh, _scale, vl);
                            int64_t _v = float2int8(_vlow, _vhigh);
                            *(int64_t*)outptr = _v;
                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outh; i++)
                    {
                        const __fp16* ptr0 = bottom_blob.row<const __fp16>(i * 2);
                        const __fp16* ptr1 = bottom_blob.row<const __fp16>(i * 2 + 1);
                        signed char* outptr = top_blob.row<signed char>(i);

                        vl = 4;
                        vfloat32m1_t _scale0 = vle32_v_f32m1((const float*)scale_data + i * 8, vl);
                        vfloat32m1_t _scale1 = vle32_v_f32m1((const float*)scale_data + i * 8 + 4, vl);

                        for (int j = 0; j < w; j++)
                        {
                            vfloat32m1_t _vlow = vget_v_f32m2_f32m1(vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr0, vl), vl), 0);
                            vfloat32m1_t _vhigh = vget_v_f32m2_f32m1(vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr1, vl), vl), 0);
                            _vlow = vfmul_vv_f32m1(_vlow, _scale0, vl);
                            _vhigh = vfmul_vv_f32m1(_vhigh, _scale1, vl);
                            int64_t _v = float2int8(_vlow, _vhigh);
                            *(int64_t*)outptr = _v;

                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
            }
            if (out_elempack == 1)
            {
                if (scale_data_size == 1)
                {
                    const float scale = scale_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < h; i++)
                    {
                        const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
                        signed char* outptr0 = top_blob.row<signed char>(i * 4);
                        signed char* outptr1 = top_blob.row<signed char>(i * 4 + 1);
                        signed char* outptr2 = top_blob.row<signed char>(i * 4 + 2);
                        signed char* outptr3 = top_blob.row<signed char>(i * 4 + 3);

                        for (int j = 0; j < w; j++)
                        {
                            outptr0[0] = float2int8((float)ptr0[0] * scale);
                            outptr1[0] = float2int8((float)ptr0[1] * scale);
                            outptr2[0] = float2int8((float)ptr0[2] * scale);
                            outptr3[0] = float2int8((float)ptr0[3] * scale);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < h; i++)
                    {
                        const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
                        signed char* outptr0 = top_blob.row<signed char>(i * 4);
                        signed char* outptr1 = top_blob.row<signed char>(i * 4 + 1);
                        signed char* outptr2 = top_blob.row<signed char>(i * 4 + 2);
                        signed char* outptr3 = top_blob.row<signed char>(i * 4 + 3);

                        const float s0 = scale_data[i * 4];
                        const float s1 = scale_data[i * 4 + 1];
                        const float s2 = scale_data[i * 4 + 2];
                        const float s3 = scale_data[i * 4 + 3];

                        for (int j = 0; j < w; j++)
                        {
                            outptr0[0] = float2int8((float)ptr0[0] * s0);
                            outptr1[0] = float2int8((float)ptr0[1] * s1);
                            outptr2[0] = float2int8((float)ptr0[2] * s2);
                            outptr3[0] = float2int8((float)ptr0[3] * s3);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;
            int out_elempack = opt.use_packing_layout && channels * elempack % 8 == 0 ? 8 : 1;
            int outc = channels * elempack / out_elempack;

            top_blob.create(w, h, outc, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (out_elempack == 8)
            {
                if (scale_data_size == 1)
                {
                    float _scale = scale_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < outc; q++)
                    {
                        const __fp16* ptr0 = bottom_blob.channel(q * 2);
                        const __fp16* ptr1 = bottom_blob.channel(q * 2 + 1);
                        signed char* outptr = top_blob.channel(q);

                        vl = 4;

                        for (int i = 0; i < size; i++)
                        {
                            vfloat32m1_t _vlow = vget_v_f32m2_f32m1(vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr0, vl), vl), 0);
                            vfloat32m1_t _vhigh = vget_v_f32m2_f32m1(vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr1, vl), vl), 0);
                            _vlow = vfmul_vf_f32m1(_vlow, _scale, vl);
                            _vhigh = vfmul_vf_f32m1(_vhigh, _scale, vl);
                            int64_t _v = float2int8(_vlow, _vhigh);
                            *(int64_t*)outptr = _v;

                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < outc; q++)
                    {
                        const __fp16* ptr0 = bottom_blob.channel(q * 2);
                        const __fp16* ptr1 = bottom_blob.channel(q * 2 + 1);
                        signed char* outptr = top_blob.channel(q);

                        vl = 4;
                        vfloat32m1_t _scale0 = vle32_v_f32m1((const float*)scale_data + q * 8, vl);
                        vfloat32m1_t _scale1 = vle32_v_f32m1((const float*)scale_data + q * 8 + 4, vl);

                        for (int i = 0; i < size; i++)
                        {
                            vfloat32m1_t _vlow = vget_v_f32m2_f32m1(vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr0, vl), vl), 0);
                            vfloat32m1_t _vhigh = vget_v_f32m2_f32m1(vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr1, vl), vl), 0);
                            _vlow = vfmul_vv_f32m1(_vlow, _scale0, vl);
                            _vhigh = vfmul_vv_f32m1(_vhigh, _scale1, vl);
                            int64_t _v = float2int8(_vlow, _vhigh);
                            *(int64_t*)outptr = _v;

                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
            }
            if (out_elempack == 1)
            {
                if (scale_data_size == 1)
                {
                    const float scale = scale_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const __fp16* ptr0 = bottom_blob.channel(q);
                        signed char* outptr0 = top_blob.channel(q * 4);
                        signed char* outptr1 = top_blob.channel(q * 4 + 1);
                        signed char* outptr2 = top_blob.channel(q * 4 + 2);
                        signed char* outptr3 = top_blob.channel(q * 4 + 3);

                        for (int i = 0; i < size; i++)
                        {
                            outptr0[0] = float2int8((float)ptr0[0] * scale);
                            outptr1[0] = float2int8((float)ptr0[1] * scale);
                            outptr2[0] = float2int8((float)ptr0[2] * scale);
                            outptr3[0] = float2int8((float)ptr0[3] * scale);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const __fp16* ptr0 = bottom_blob.channel(q);
                        signed char* outptr0 = top_blob.channel(q * 4);
                        signed char* outptr1 = top_blob.channel(q * 4 + 1);
                        signed char* outptr2 = top_blob.channel(q * 4 + 2);
                        signed char* outptr3 = top_blob.channel(q * 4 + 3);

                        const float s0 = scale_data[q * 4];
                        const float s1 = scale_data[q * 4 + 1];
                        const float s2 = scale_data[q * 4 + 2];
                        const float s3 = scale_data[q * 4 + 3];

                        for (int i = 0; i < size; i++)
                        {
                            outptr0[0] = float2int8((float)ptr0[0] * s0);
                            outptr1[0] = float2int8((float)ptr0[1] * s1);
                            outptr2[0] = float2int8((float)ptr0[2] * s2);
                            outptr3[0] = float2int8((float)ptr0[3] * s3);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
            }
        }

        return 0;
    }

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const __fp16* ptr = bottom_blob;
        signed char* outptr = top_blob;

        if (scale_data_size == 1)
        {
            const float scale = scale_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                outptr[i] = float2int8((float)ptr[i] * scale);
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                outptr[i] = float2int8((float)ptr[i] * scale_data[i]);
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
            signed char* outptr0 = top_blob.row<signed char>(i);

            const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

            for (int j = 0; j < w; j++)
            {
                *outptr0++ = float2int8((float)*ptr0++ * scale);
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(w, h, channels, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = bottom_blob.channel(q);
            signed char* outptr = top_blob.channel(q);

            const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

            for (int i = 0; i < size; i++)
            {
                *outptr++ = float2int8((float)*ptr++ * scale);
            }
        }
    }

    return 0;
}

int Quantize_riscv::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;
    int vl;

    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;

            top_blob.create(w, (size_t)8u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                __fp16 _scale = (__fp16)scale_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const __fp16* ptr0 = (const __fp16*)bottom_blob + i * 8;
                    signed char* outptr = (signed char*)top_blob + i * 8;
                    vl = 8;
                    vfloat16m1_t _v = vle16_v_f16m1(ptr0, vl);
                    _v = vfmul_vf_f16m1(_v, _scale, vl);
                    *(int64_t*)outptr = float2int8(_v);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const __fp16* ptr0 = (const __fp16*)bottom_blob + i * 8;
                    signed char* outptr = (signed char*)top_blob + i * 8;

                    vl = 8;
                    vfloat16m1_t _v = vle16_v_f16m1(ptr0, vl);
                    vfloat16m1_t _scale = vfncvt_f_f_w_f16m1(vle32_v_f32m2((const float*)scale_data + i * 8, vl), vl);

                    _v = vfmul_vv_f16m1(_v, _scale, vl);
                    *(int64_t*)outptr = float2int8(_v);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;

            top_blob.create(w, h, (size_t)8u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                __fp16 _scale = (__fp16)scale_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
                    signed char* outptr0 = top_blob.row<signed char>(i);

                    for (int j = 0; j < w; j++)
                    {
                        vl = 8;
                        vfloat16m1_t _v = vle16_v_f16m1(ptr0, vl);
                        _v = vfmul_vf_f16m1(_v, _scale, vl);

                        *(int64_t*)outptr0 = float2int8(_v);

                        ptr0 += 8;
                        outptr0 += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
                    signed char* outptr0 = top_blob.row<signed char>(i);

                    vl = 8;
                    vfloat16m1_t _scale = vfncvt_f_f_w_f16m1(vle32_v_f32m2((const float*)scale_data + i * 8, vl), vl);

                    for (int j = 0; j < w; j++)
                    {
                        vfloat16m1_t _v = vle16_v_f16m1(ptr0, vl);
                        _v = vfmul_vv_f16m1(_v, _scale, vl);
                        *(int64_t*)outptr0 = float2int8(_v);

                        ptr0 += 8;
                        outptr0 += 8;
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;

            top_blob.create(w, h, channels, (size_t)8u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                __fp16 _scale = (__fp16)scale_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr0 = bottom_blob.channel(q);
                    signed char* outptr0 = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        vl = 8;
                        vfloat16m1_t _v = vle16_v_f16m1(ptr0, vl);
                        _v = vfmul_vf_f16m1(_v, _scale, vl);
                        *(int64_t*)outptr0 = float2int8(_v);

                        ptr0 += 8;
                        outptr0 += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr0 = bottom_blob.channel(q);
                    signed char* outptr0 = top_blob.channel(q);

                    vl = 8;
                    vfloat16m1_t _scale = vfncvt_f_f_w_f16m1(vle32_v_f32m2((const float*)scale_data + q * 8, vl), vl);

                    for (int i = 0; i < size; i++)
                    {
                        vfloat16m1_t _v = vle16_v_f16m1(ptr0, vl);
                        _v = vfmul_vv_f16m1(_v, _scale, vl);
                        *(int64_t*)outptr0 = float2int8(_v);

                        ptr0 += 8;
                        outptr0 += 8;
                    }
                }
            }
        }

        return 0;
    }

    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int outw = w * elempack;

            top_blob.create(outw, (size_t)1u, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                const __fp16 scale = scale_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const __fp16* ptr0 = (const __fp16*)bottom_blob + i * 4;
                    signed char* outptr = (signed char*)top_blob + i * 4;

                    outptr[0] = float2int8(ptr0[0] * scale);
                    outptr[1] = float2int8(ptr0[1] * scale);
                    outptr[2] = float2int8(ptr0[2] * scale);
                    outptr[3] = float2int8(ptr0[3] * scale);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const __fp16* ptr0 = (const __fp16*)bottom_blob + i * 4;
                    signed char* outptr = (signed char*)top_blob + i * 4;

                    outptr[0] = float2int8(ptr0[0] * (__fp16)scale_data[i * 4]);
                    outptr[1] = float2int8(ptr0[1] * (__fp16)scale_data[i * 4 + 1]);
                    outptr[2] = float2int8(ptr0[2] * (__fp16)scale_data[i * 4 + 2]);
                    outptr[3] = float2int8(ptr0[3] * (__fp16)scale_data[i * 4 + 3]);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int outh = h * elempack;

            top_blob.create(w, outh, (size_t)1u, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                const __fp16 scale = scale_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
                    signed char* outptr0 = top_blob.row<signed char>(i * 4);
                    signed char* outptr1 = top_blob.row<signed char>(i * 4 + 1);
                    signed char* outptr2 = top_blob.row<signed char>(i * 4 + 2);
                    signed char* outptr3 = top_blob.row<signed char>(i * 4 + 3);

                    for (int j = 0; j < w; j++)
                    {
                        outptr0[0] = float2int8(ptr0[0] * scale);
                        outptr1[0] = float2int8(ptr0[1] * scale);
                        outptr2[0] = float2int8(ptr0[2] * scale);
                        outptr3[0] = float2int8(ptr0[3] * scale);

                        ptr0 += 4;
                        outptr0 += 1;
                        outptr1 += 1;
                        outptr2 += 1;
                        outptr3 += 1;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
                    signed char* outptr0 = top_blob.row<signed char>(i * 4);
                    signed char* outptr1 = top_blob.row<signed char>(i * 4 + 1);
                    signed char* outptr2 = top_blob.row<signed char>(i * 4 + 2);
                    signed char* outptr3 = top_blob.row<signed char>(i * 4 + 3);

                    const __fp16 s0 = scale_data[i * 4];
                    const __fp16 s1 = scale_data[i * 4 + 1];
                    const __fp16 s2 = scale_data[i * 4 + 2];
                    const __fp16 s3 = scale_data[i * 4 + 3];

                    for (int j = 0; j < w; j++)
                    {
                        outptr0[0] = float2int8(ptr0[0] * s0);
                        outptr1[0] = float2int8(ptr0[1] * s1);
                        outptr2[0] = float2int8(ptr0[2] * s2);
                        outptr3[0] = float2int8(ptr0[3] * s3);

                        ptr0 += 4;
                        outptr0 += 1;
                        outptr1 += 1;
                        outptr2 += 1;
                        outptr3 += 1;
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;
            int outc = channels * elempack;

            top_blob.create(w, h, outc, (size_t)1u, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                const __fp16 scale = scale_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr0 = bottom_blob.channel(q);
                    signed char* outptr0 = top_blob.channel(q * 4);
                    signed char* outptr1 = top_blob.channel(q * 4 + 1);
                    signed char* outptr2 = top_blob.channel(q * 4 + 2);
                    signed char* outptr3 = top_blob.channel(q * 4 + 3);

                    for (int i = 0; i < size; i++)
                    {
                        outptr0[0] = float2int8(ptr0[0] * scale);
                        outptr1[0] = float2int8(ptr0[1] * scale);
                        outptr2[0] = float2int8(ptr0[2] * scale);
                        outptr3[0] = float2int8(ptr0[3] * scale);

                        ptr0 += 4;
                        outptr0 += 1;
                        outptr1 += 1;
                        outptr2 += 1;
                        outptr3 += 1;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr0 = bottom_blob.channel(q);
                    signed char* outptr0 = top_blob.channel(q * 4);
                    signed char* outptr1 = top_blob.channel(q * 4 + 1);
                    signed char* outptr2 = top_blob.channel(q * 4 + 2);
                    signed char* outptr3 = top_blob.channel(q * 4 + 3);

                    const __fp16 s0 = scale_data[q * 4];
                    const __fp16 s1 = scale_data[q * 4 + 1];
                    const __fp16 s2 = scale_data[q * 4 + 2];
                    const __fp16 s3 = scale_data[q * 4 + 3];

                    for (int i = 0; i < size; i++)
                    {
                        outptr0[0] = float2int8(ptr0[0] * s0);
                        outptr1[0] = float2int8(ptr0[1] * s1);
                        outptr2[0] = float2int8(ptr0[2] * s2);
                        outptr3[0] = float2int8(ptr0[3] * s3);

                        ptr0 += 4;
                        outptr0 += 1;
                        outptr1 += 1;
                        outptr2 += 1;
                        outptr3 += 1;
                    }
                }
            }
        }

        return 0;
    }

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const __fp16* ptr = bottom_blob;
        signed char* outptr = top_blob;

        if (scale_data_size == 1)
        {
            const __fp16 scale = scale_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                outptr[i] = float2int8(ptr[i] * scale);
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                outptr[i] = float2int8(ptr[i] * (__fp16)scale_data[i]);
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
            signed char* outptr0 = top_blob.row<signed char>(i);

            const __fp16 scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

            for (int j = 0; j < w; j++)
            {
                *outptr0++ = float2int8(*ptr0++ * scale);
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(w, h, channels, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = bottom_blob.channel(q);
            signed char* outptr = top_blob.channel(q);

            const __fp16 scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

            for (int i = 0; i < size; i++)
            {
                *outptr++ = float2int8(*ptr++ * scale);
            }
        }
    }

    return 0;
}

#endif // __riscv_vector && __riscv_zfh

} // namespace ncnn
