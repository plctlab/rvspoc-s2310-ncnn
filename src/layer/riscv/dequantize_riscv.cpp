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

#include "dequantize_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"
#include "cpu.h"

namespace ncnn {

Dequantize_riscv::Dequantize_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif // __riscv_zfh
#endif // __riscv_vector
}

int Dequantize_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int vl;
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

    if (support_fp16_storage && opt.use_fp16_storage)
    {
        if (opt.use_fp16_arithmetic)
        {
            return forward_fp16sa(bottom_blob, top_blob, opt);
        }
        else
        {
            return forward_fp16s(bottom_blob, top_blob, opt);
        }
    }

#if __riscv_vector
    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int outw = w * 2;

            top_blob.create(outw, (size_t)16u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                float _scale = scale_data[0];
                if (bias_data_size == 0)
                {
                    // #pragma omp parallel for num_threads(opt.num_threads)
                    int n = outw * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        float* ptr = (float*)top_blob + offset;
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmul_vf_f32m8(_v, _scale, vl);
                        vse32_v_f32m8(ptr, _v, vl);
                        offset += vl;
                        n -= vl;
                    }
                }
                else if (bias_data_size == 1)
                {
                    int n = outw * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        float* ptr = (float*)top_blob + offset;
                        vfloat32m8_t _bias = vfmv_v_f_f32m8(bias_data[0], vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmadd_vf_f32m8(_v, _scale, _bias, vl);
                        vse32_v_f32m8(ptr, _v, vl);
                        offset += vl;
                        n -= vl;
                    }
                }
                else
                {
                    // #pragma omp parallel for num_threads(opt.num_threads)
                    int n = outw * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        float* ptr = (float*)top_blob + offset;
                        vfloat32m8_t _bias = vle32_v_f32m8((const float*)bias_data + offset, vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmadd_vf_f32m8(_v, _scale, _bias, vl);
                        vse32_v_f32m8(ptr, _v, vl);
                        offset += vl;
                        n -= vl;
                    }
                }
            }
            else
            {
                if (bias_data_size == 0)
                {
                    // #pragma omp parallel for num_threads(opt.num_threads)
                    int n = outw * 4;
                    int offset = 0;

                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        float* ptr = (float*)top_blob + offset;

                        vfloat32m8_t _scale = vle32_v_f32m8((const float*)scale_data + offset, vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmul_vv_f32m8(_v, _scale, vl);
                        vse32_v_f32m8(ptr, _v, vl);

                        offset += vl;
                        n -= vl;
                    }
                }
                else if (bias_data_size == 1)
                {
                    int n = outw * 4;
                    int offset = 0;

                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        float* ptr = (float*)top_blob + offset;

                        vfloat32m8_t _scale = vle32_v_f32m8((const float*)scale_data + offset, vl);
                        vfloat32m8_t _bias = vfmv_v_f_f32m8(bias_data[0], vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                        vse32_v_f32m8(ptr, _v, vl);
                        offset += vl;
                        n -= vl;
                    }
                }
                else
                {
                    int n = outw * 4;
                    int offset = 0;

                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        float* ptr = (float*)top_blob + offset;

                        vfloat32m8_t _scale = vle32_v_f32m8((const float*)scale_data + offset, vl);
                        vfloat32m8_t _bias = vle32_v_f32m8((const float*)bias_data + offset, vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                        vse32_v_f32m8(ptr, _v, vl);
                        offset += vl;
                        n -= vl;
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int outh = h * 2;
            vl = 4;

            top_blob.create(w, outh, (size_t)16u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr0 = top_blob.row(i * 2);
                    float* ptr1 = top_blob.row(i * 2 + 1);

                    vl = 4;
                    vfloat32m1_t _scale0 = scale_data_size == 1 ? vfmv_v_f_f32m1(scale_data[0], vl) : vle32_v_f32m1((const float*)scale_data + i * 8, vl);
                    vfloat32m1_t _scale1 = scale_data_size == 1 ? vfmv_v_f_f32m1(scale_data[0], vl) : vle32_v_f32m1((const float*)scale_data + i * 8 + 4, vl);

                    for (int j = 0; j < w; j++)
                    {
                        vfloat32m1_t _v0 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr, vl), vl);
                        vfloat32m1_t _v1 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr + 4, vl), vl);
                        _v0 = vfmul_vv_f32m1(_v0, _scale0, vl);
                        _v1 = vfmul_vv_f32m1(_v1, _scale1, vl);
                        vse32_v_f32m1(ptr0, _v0, vl);
                        vse32_v_f32m1(ptr1, _v1, vl);
                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr0 = top_blob.row(i * 2);
                    float* ptr1 = top_blob.row(i * 2 + 1);

                    vl = 4;
                    vfloat32m1_t _scale0 = scale_data_size == 1 ? vfmv_v_f_f32m1(scale_data[0], vl) : vle32_v_f32m1((const float*)scale_data + i * 8, vl);
                    vfloat32m1_t _scale1 = scale_data_size == 1 ? vfmv_v_f_f32m1(scale_data[0], vl) : vle32_v_f32m1((const float*)scale_data + i * 8 + 4, vl);
                    vfloat32m1_t _bias0 = bias_data_size == 1 ? vfmv_v_f_f32m1(bias_data[0], vl) : vle32_v_f32m1((const float*)bias_data + i * 8, vl);
                    vfloat32m1_t _bias1 = bias_data_size == 1 ? vfmv_v_f_f32m1(bias_data[0], vl) : vle32_v_f32m1((const float*)bias_data + i * 8 + 4, vl);

                    for (int j = 0; j < w; j++)
                    {
                        vfloat32m1_t _v0 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr, vl), vl);
                        vfloat32m1_t _v1 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr + 4, vl), vl);
                        _v0 = vfmadd_vv_f32m1(_v0, _scale0, _bias0, vl);
                        _v1 = vfmadd_vv_f32m1(_v1, _scale1, _bias1, vl);
                        vse32_v_f32m1(ptr0, _v0, vl);
                        vse32_v_f32m1(ptr1, _v1, vl);
                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
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
            int outc = channels * 2;

            top_blob.create(w, h, outc, (size_t)16u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr0 = top_blob.channel(q * 2);
                    float* ptr1 = top_blob.channel(q * 2 + 1);

                    vl = 4;
                    vfloat32m1_t _scale0 = scale_data_size == 1 ? vfmv_v_f_f32m1(scale_data[0], vl) : vle32_v_f32m1((const float*)scale_data + q * 8, vl);
                    vfloat32m1_t _scale1 = scale_data_size == 1 ? vfmv_v_f_f32m1(scale_data[0], vl) : vle32_v_f32m1((const float*)scale_data + q * 8 + 4, vl);

                    int i = 0;
                    for (; i + 1 < size; i += 2)
                    {
                        vfloat32m1_t _v0 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr, vl), vl);
                        vfloat32m1_t _v1 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr + 4, vl), vl);
                        vfloat32m1_t _v2 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr + 8, vl), vl);
                        vfloat32m1_t _v3 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr + 12, vl), vl);

                        _v0 = vfmul_vv_f32m1(_v0, _scale0, vl);
                        _v1 = vfmul_vv_f32m1(_v1, _scale1, vl);
                        _v2 = vfmul_vv_f32m1(_v2, _scale0, vl);
                        _v3 = vfmul_vv_f32m1(_v3, _scale1, vl);

                        vse32_v_f32m1(ptr0, _v0, vl);
                        vse32_v_f32m1(ptr0 + 4, _v2, vl);
                        vse32_v_f32m1(ptr1, _v1, vl);
                        vse32_v_f32m1(ptr1 + 4, _v3, vl);
                        intptr += 16;
                        ptr0 += 8;
                        ptr1 += 8;
                    }
                    for (; i < size; i++)
                    {
                        vfloat32m1_t _v0 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr, vl), vl);
                        vfloat32m1_t _v1 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr + 4, vl), vl);

                        _v0 = vfmul_vv_f32m1(_v0, _scale0, vl);
                        _v1 = vfmul_vv_f32m1(_v1, _scale1, vl);

                        vse32_v_f32m1(ptr0, _v0, vl);
                        vse32_v_f32m1(ptr1, _v1, vl);

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr0 = top_blob.channel(q * 2);
                    float* ptr1 = top_blob.channel(q * 2 + 1);
                    vl = 4;

                    vfloat32m1_t _scale0 = scale_data_size == 1 ? vfmv_v_f_f32m1(scale_data[0], vl) : vle32_v_f32m1((const float*)scale_data + q * 8, vl);
                    vfloat32m1_t _scale1 = scale_data_size == 1 ? vfmv_v_f_f32m1(scale_data[0], vl) : vle32_v_f32m1((const float*)scale_data + q * 8 + 4, vl);
                    vfloat32m1_t _bias0 = bias_data_size == 1 ? vfmv_v_f_f32m1(bias_data[0], vl) : vle32_v_f32m1((const float*)bias_data + q * 8, vl);
                    vfloat32m1_t _bias1 = bias_data_size == 1 ? vfmv_v_f_f32m1(bias_data[0], vl) : vle32_v_f32m1((const float*)bias_data + q * 8 + 4, vl);

                    int i = 0;
                    for (; i + 1 < size; i += 2)
                    {
                        vfloat32m1_t _v0 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr, vl), vl);
                        vfloat32m1_t _v1 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr + 4, vl), vl);
                        vfloat32m1_t _v2 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr + 8, vl), vl);
                        vfloat32m1_t _v3 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr + 12, vl), vl);

                        _v0 = vfmadd_vv_f32m1(_v0, _scale0, _bias0, vl);
                        _v1 = vfmadd_vv_f32m1(_v1, _scale1, _bias1, vl);
                        _v2 = vfmadd_vv_f32m1(_v2, _scale0, _bias0, vl);
                        _v3 = vfmadd_vv_f32m1(_v3, _scale1, _bias1, vl);

                        vse32_v_f32m1(ptr0, _v0, vl);
                        vse32_v_f32m1(ptr0 + 4, _v2, vl);
                        vse32_v_f32m1(ptr1, _v1, vl);
                        vse32_v_f32m1(ptr1 + 4, _v3, vl);

                        intptr += 16;
                        ptr0 += 8;
                        ptr1 += 8;
                    }
                    for (; i < size; i++)
                    {
                        vfloat32m1_t _v0 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr, vl), vl);
                        vfloat32m1_t _v1 = vfcvt_f_x_v_f32m1(vle32_v_i32m1(intptr + 4, vl), vl);

                        _v0 = vfmadd_vv_f32m1(_v0, _scale0, _bias0, vl);
                        _v1 = vfmadd_vv_f32m1(_v1, _scale1, _bias1, vl);

                        vse32_v_f32m1(ptr0, _v0, vl);
                        vse32_v_f32m1(ptr1, _v1, vl);

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
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

            top_blob.create(w, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                if (bias_data_size == 0)
                {
                    int n = w * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        float* ptr = (float*)top_blob + offset;
                        vfloat32m8_t _scale = vfmv_v_f_f32m8(scale_data[0], vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmul_vv_f32m8(_v, _scale, vl);
                        vse32_v_f32m8(ptr, _v, vl);
                        offset += vl;
                        n -= vl;
                    }
                }
                else if (bias_data_size == 1)
                {
                    int n = w * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        float* ptr = (float*)top_blob + offset;
                        vfloat32m8_t _scale = vfmv_v_f_f32m8(scale_data[0], vl);
                        vfloat32m8_t _bias = vfmv_v_f_f32m8(bias_data[0], vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                        vse32_v_f32m8(ptr, _v, vl);
                        offset += vl;
                        n -= vl;
                    }
                }
                else
                {
                    // #pragma omp parallel for num_threads(opt.num_threads)
                    int n = w * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        float* ptr = (float*)top_blob + offset;
                        vfloat32m8_t _scale = vfmv_v_f_f32m8(scale_data[0], vl);
                        vfloat32m8_t _bias = vle32_v_f32m8((const float*)bias_data + offset, vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                        vse32_v_f32m8(ptr, _v, vl);
                        offset += vl;
                        n -= vl;
                    }
                }
            }
            else
            {
                if (bias_data_size == 0)
                {
                    // #pragma omp parallel for num_threads(opt.num_threads)
                    int n = w * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        float* ptr = (float*)top_blob + offset;
                        vfloat32m8_t _scale = vle32_v_f32m8((const float*)scale_data + offset, vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmul_vv_f32m8(_v, _scale, vl);
                        vse32_v_f32m8(ptr, _v, vl);
                        offset += vl;
                        n -= vl;
                    }
                }
                else if (bias_data_size == 1)
                {
                    int n = w * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        float* ptr = (float*)top_blob + offset;
                        vfloat32m8_t _scale = vle32_v_f32m8((const float*)scale_data + offset, vl);
                        vfloat32m8_t _bias = vfmv_v_f_f32m8(bias_data[0], vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                        vse32_v_f32m8(ptr, _v, vl);
                        offset += vl;
                        n -= vl;
                    }
                }
                else
                {
                    int n = w * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        float* ptr = (float*)top_blob + offset;
                        vfloat32m8_t _scale = vle32_v_f32m8((const float*)scale_data + offset, vl);
                        vfloat32m8_t _bias = vle32_v_f32m8((const float*)bias_data + offset, vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                        vse32_v_f32m8(ptr, _v, vl);
                        offset += vl;
                        n -= vl;
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;

            top_blob.create(w, h, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr = top_blob.row(i);

                    int n = w * 4;
                    int offset = 0;
                    vl = 4;
                    vfloat32m1_t _scale_m1 = scale_data_size == 1 ? vfmv_v_f_f32m1(scale_data[0], vl) : vle32_v_f32m1((const float*)scale_data + i * 4, vl);
                    vfloat32m8_t _scale = vundefined_f32m8();
                    _scale = vset_v_f32m1_f32m8(_scale, 0, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 1, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 2, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 3, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 4, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 5, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 6, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 7, _scale_m1);

                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr + offset, vl), vl);
                        _v = vfmul_vv_f32m8(_v, _scale, vl);
                        vse32_v_f32m8(ptr + offset, _v, vl);
                        offset += vl;
                        n -= vl;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr = top_blob.row(i);

                    vl = 4;
                    vfloat32m1_t _scale_m1 = scale_data_size == 1 ? vfmv_v_f_f32m1(scale_data[0], vl) : vle32_v_f32m1((const float*)scale_data + i * 4, vl);
                    vfloat32m1_t _bias_m1 = bias_data_size == 1 ? vfmv_v_f_f32m1(bias_data[0], vl) : vle32_v_f32m1((const float*)bias_data + i * 4, vl);

                    vfloat32m8_t _scale = vundefined_f32m8();
                    _scale = vset_v_f32m1_f32m8(_scale, 0, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 1, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 2, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 3, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 4, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 5, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 6, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 7, _scale_m1);

                    vfloat32m8_t _bias = vundefined_f32m8();
                    _bias = vset_v_f32m1_f32m8(_bias, 0, _bias_m1);
                    _bias = vset_v_f32m1_f32m8(_bias, 1, _bias_m1);
                    _bias = vset_v_f32m1_f32m8(_bias, 2, _bias_m1);
                    _bias = vset_v_f32m1_f32m8(_bias, 3, _bias_m1);
                    _bias = vset_v_f32m1_f32m8(_bias, 4, _bias_m1);
                    _bias = vset_v_f32m1_f32m8(_bias, 5, _bias_m1);
                    _bias = vset_v_f32m1_f32m8(_bias, 6, _bias_m1);
                    _bias = vset_v_f32m1_f32m8(_bias, 7, _bias_m1);

                    int n = w * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr + offset, vl), vl);
                        _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                        vse32_v_f32m8(ptr + offset, _v, vl);
                        offset += vl;
                        n -= vl;
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

            top_blob.create(w, h, channels, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr = top_blob.channel(q);
                    vl = 4;
                    vfloat32m1_t _scale_m1 = scale_data_size == 1 ? vfmv_v_f_f32m1(scale_data[0], vl) : vle32_v_f32m1((const float*)scale_data + q * 4, vl);
                    vfloat32m8_t _scale = vundefined_f32m8();
                    _scale = vset_v_f32m1_f32m8(_scale, 0, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 1, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 2, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 3, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 4, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 5, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 6, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 7, _scale_m1);

                    int n = size * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr + offset, vl), vl);
                        _v = vfmul_vv_f32m8(_v, _scale, vl);
                        vse32_v_f32m8(ptr + offset, _v, vl);
                        offset += vl;
                        n -= vl;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr = top_blob.channel(q);

                    vl = 4;
                    vfloat32m1_t _scale_m1 = scale_data_size == 1 ? vfmv_v_f_f32m1(scale_data[0], vl) : vle32_v_f32m1((const float*)scale_data + q * 4, vl);
                    vfloat32m1_t _bias_m1 = bias_data_size == 1 ? vfmv_v_f_f32m1(bias_data[0], vl) : vle32_v_f32m1((const float*)bias_data + q * 4, vl);

                    vfloat32m8_t _scale = vundefined_f32m8();
                    _scale = vset_v_f32m1_f32m8(_scale, 0, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 1, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 2, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 3, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 4, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 5, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 6, _scale_m1);
                    _scale = vset_v_f32m1_f32m8(_scale, 7, _scale_m1);

                    vfloat32m8_t _bias = vundefined_f32m8();
                    _bias = vset_v_f32m1_f32m8(_bias, 0, _bias_m1);
                    _bias = vset_v_f32m1_f32m8(_bias, 1, _bias_m1);
                    _bias = vset_v_f32m1_f32m8(_bias, 2, _bias_m1);
                    _bias = vset_v_f32m1_f32m8(_bias, 3, _bias_m1);
                    _bias = vset_v_f32m1_f32m8(_bias, 4, _bias_m1);
                    _bias = vset_v_f32m1_f32m8(_bias, 5, _bias_m1);
                    _bias = vset_v_f32m1_f32m8(_bias, 6, _bias_m1);
                    _bias = vset_v_f32m1_f32m8(_bias, 7, _bias_m1);

                    int n = size * 4;
                    int offset = 0;

                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr + offset, vl), vl);
                        _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                        vse32_v_f32m8(ptr + offset, _v, vl);
                        offset += vl;
                        n -= vl;
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

        top_blob.create(w, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int* intptr = bottom_blob;
        float* ptr = top_blob;

        if (scale_data_size == 1)
        {
            const float scale = scale_data[0];

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale;
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale + bias;
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale + bias_data[i];
                }
            }
        }
        else
        {
            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i];
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i] + bias;
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i] + bias_data[i];
                }
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                float* ptr = top_blob.row(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

                int n = w;
                int offset = 0;
                while (n > 0)
                {
                    vl = vsetvl_e32m8(n);
                    vfloat32m8_t _scale = vfmv_v_f_f32m8(scale, vl);
                    vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr + offset, vl), vl);
                    _v = vfmul_vv_f32m8(_v, _scale, vl);
                    vse32_v_f32m8(ptr + offset, _v, vl);
                    offset += vl;
                    n -= vl;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                float* ptr = top_blob.row(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[i];

                int n = w;
                int offset = 0;
                while (n > 0)
                {
                    vl = vsetvl_e32m8(n);
                    vfloat32m8_t _scale = vfmv_v_f_f32m8(scale, vl);
                    vfloat32m8_t _bias = vfmv_v_f_f32m8(bias, vl);
                    vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr + offset, vl), vl);
                    _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                    vse32_v_f32m8(ptr + offset, _v, vl);
                    offset += vl;
                    n -= vl;
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

        top_blob.create(w, h, channels, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                float* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

                int i = 0;
#if __riscv_vector
                int n = size;
                int offset = 0;

                while (n > 0)
                {
                    vl = vsetvl_e32m8(n);
                    vfloat32m8_t _scale = vfmv_v_f_f32m8(scale, vl);
                    vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr + offset, vl), vl);
                    _v = vfmul_vv_f32m8(_v, _scale, vl);
                    vse32_v_f32m8(ptr + offset, _v, vl);
                    offset += vl;
                    n -= vl;
                }
#endif // __riscv_vector               \
// for (; i < size; i++)           \
// {                               \
//     *ptr++ = *intptr++ * scale; \
// }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                float* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[q];

                int i = 0;
#if __riscv_vector
                int n = size;
                int offset = 0;

                while (n > 0)
                {
                    vl = vsetvl_e32m8(n);
                    vfloat32m8_t _scale = vfmv_v_f_f32m8(scale, vl);
                    vfloat32m8_t _bias = vfmv_v_f_f32m8(bias, vl);
                    vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr + offset, vl), vl);
                    _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                    vse32_v_f32m8(ptr + offset, _v, vl);
                    offset += vl;
                    n -= vl;
                }
#endif // __riscv_vector                      \
// for (; i < size; i++)                  \
// {                                      \
//     *ptr++ = *intptr++ * scale + bias; \
// }
            }
        }
    }

    return 0;
}

#if __riscv_vector && __riscv_zfh

int Dequantize_riscv::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;
    int vl;

    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int outw = w * 2;

            top_blob.create(outw, (size_t)8u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                // float32x4_t _scale = vdupq_n_f32(scale_data[0]);
                float _scale = scale_data[0];

                if (bias_data_size == 0)
                {
                    // #pragma omp parallel for num_threads(opt.num_threads)
                    // for (int i = 0; i < outw; i++)
                    // {
                    //     const int* intptr = (const int*)bottom_blob + i * 4;
                    //     __fp16* ptr = (__fp16*)top_blob + i * 4;
                    //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     _v = vmulq_f32(_v, _scale);
                    //     vst1_f16(ptr, vcvt_f16_f32(_v));
                    // }
                    int n = outw * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        __fp16* ptr = (__fp16*)top_blob + offset;
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmul_vf_f32m8(_v, _scale, vl);
                        // vse32_v_f32m8(ptr, _v, vl);
                        vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_v, vl), vl);
                        offset += vl;
                        n -= vl;
                    }
                }
                else if (bias_data_size == 1)
                {
                    // float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    // #pragma omp parallel for num_threads(opt.num_threads)
                    // for (int i = 0; i < outw; i++)
                    // {
                    //     const int* intptr = (const int*)bottom_blob + i * 4;
                    //     __fp16* ptr = (__fp16*)top_blob + i * 4;

                    //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     _v = vfmaq_f32(_bias, _v, _scale);
                    //     vst1_f16(ptr, vcvt_f16_f32(_v));
                    // }
                    int n = outw * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        __fp16* ptr = (__fp16*)top_blob + offset;
                        vfloat32m8_t _bias = vfmv_v_f_f32m8(bias_data[0], vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmadd_vf_f32m8(_v, _scale, _bias, vl);
                        // vse32_v_f32m8(ptr, _v, vl);
                        vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_v, vl), vl);
                        offset += vl;
                        n -= vl;
                    }
                }
                else
                {
                    // #pragma omp parallel for num_threads(opt.num_threads)
                    // for (int i = 0; i < outw; i++)
                    // {
                    //     const int* intptr = (const int*)bottom_blob + i * 4;
                    //     __fp16* ptr = (__fp16*)top_blob + i * 4;

                    //     float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                    //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     _v = vfmaq_f32(_bias, _v, _scale);
                    //     vst1_f16(ptr, vcvt_f16_f32(_v));
                    // }
                    int n = outw * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        __fp16* ptr = (__fp16*)top_blob + offset;
                        vfloat32m8_t _bias = vle32_v_f32m8((const float*)bias_data + offset, vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmadd_vf_f32m8(_v, _scale, _bias, vl);
                        // vse32_v_f32m8(ptr, _v, vl);
                        vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_v, vl), vl);
                        offset += vl;
                        n -= vl;
                    }
                }
            }
            else
            {
                if (bias_data_size == 0)
                {
                    // #pragma omp parallel for num_threads(opt.num_threads)
                    // for (int i = 0; i < outw; i++)
                    // {
                    //     const int* intptr = (const int*)bottom_blob + i * 4;
                    //     __fp16* ptr = (__fp16*)top_blob + i * 4;

                    //     float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                    //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     _v = vmulq_f32(_v, _scale);
                    //     vst1_f16(ptr, vcvt_f16_f32(_v));
                    // }
                    int n = outw * 4;
                    int offset = 0;

                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        __fp16* ptr = (__fp16*)top_blob + offset;

                        vfloat32m8_t _scale = vle32_v_f32m8((const float*)scale_data + offset, vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmul_vv_f32m8(_v, _scale, vl);
                        // vse32_v_f32m8(ptr, _v, vl);
                        vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_v, vl), vl);
                        offset += vl;
                        n -= vl;
                    }
                }
                else if (bias_data_size == 1)
                {
                    // float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    // #pragma omp parallel for num_threads(opt.num_threads)
                    // for (int i = 0; i < outw; i++)
                    // {
                    //     const int* intptr = (const int*)bottom_blob + i * 4;
                    //     __fp16* ptr = (__fp16*)top_blob + i * 4;

                    //     float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                    //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     _v = vfmaq_f32(_bias, _v, _scale);
                    //     vst1_f16(ptr, vcvt_f16_f32(_v));
                    // }
                    int n = outw * 4;
                    int offset = 0;

                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        __fp16* ptr = (__fp16*)top_blob + offset;

                        vfloat32m8_t _scale = vle32_v_f32m8((const float*)scale_data + offset, vl);
                        vfloat32m8_t _bias = vfmv_v_f_f32m8(bias_data[0], vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                        // vse32_v_f32m8(ptr, _v, vl);
                        vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_v, vl), vl);
                        offset += vl;
                        n -= vl;
                    }
                }
                else
                {
                    // #pragma omp parallel for num_threads(opt.num_threads)
                    // for (int i = 0; i < outw; i++)
                    // {
                    //     const int* intptr = (const int*)bottom_blob + i * 4;
                    //     __fp16* ptr = (__fp16*)top_blob + i * 4;

                    //     float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                    //     float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                    //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     _v = vfmaq_f32(_bias, _v, _scale);
                    //     vst1_f16(ptr, vcvt_f16_f32(_v));
                    // }
                    int n = outw * 4;
                    int offset = 0;

                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        __fp16* ptr = (__fp16*)top_blob + offset;

                        vfloat32m8_t _scale = vle32_v_f32m8((const float*)scale_data + offset, vl);
                        vfloat32m8_t _bias = vle32_v_f32m8((const float*)bias_data + offset, vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                        // vse32_v_f32m8(ptr, _v, vl);
                        vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_v, vl), vl);
                        offset += vl;
                        n -= vl;
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int outh = h * 2;

            top_blob.create(w, outh, (size_t)8u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    __fp16* ptr0 = top_blob.row<__fp16>(i * 2);
                    __fp16* ptr1 = top_blob.row<__fp16>(i * 2 + 1);

                    // float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8);
                    // float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8 + 4);

                    // for (int j = 0; j < w; j++)
                    // {
                    //     float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                    //     _v0 = vmulq_f32(_v0, _scale0);
                    //     _v1 = vmulq_f32(_v1, _scale1);
                    //     vst1_f16(ptr0, vcvt_f16_f32(_v0));
                    //     vst1_f16(ptr1, vcvt_f16_f32(_v1));

                    //     intptr += 8;
                    //     ptr0 += 4;
                    //     ptr1 += 4;
                    // }
                    vl = 4;
                    vfloat32m2_t _scale0 = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + i * 8, vl);
                    vfloat32m2_t _scale1 = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + i * 8 + 4, vl);

                    for (int j = 0; j < w; j++)
                    {
                        vfloat32m2_t _v0 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        vfloat32m2_t _v1 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr + 4, vl), vl);
                        _v0 = vfmul_vv_f32m2(_v0, _scale0, vl);
                        _v1 = vfmul_vv_f32m2(_v1, _scale1, vl);
                        // vse32_v_f32m1(ptr0, _v0, vl);
                        // vse32_v_f32m1(ptr1, _v1, vl);
                        vse16_v_f16m1(ptr0, vfncvt_f_f_w_f16m1(_v0, vl), vl);
                        vse16_v_f16m1(ptr1, vfncvt_f_f_w_f16m1(_v1, vl), vl);
                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    __fp16* ptr0 = top_blob.row<__fp16>(i * 2);
                    __fp16* ptr1 = top_blob.row<__fp16>(i * 2 + 1);

                    // float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8);
                    // float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8 + 4);
                    // float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8);
                    // float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8 + 4);

                    // for (int j = 0; j < w; j++)
                    // {
                    //     float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                    //     _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                    //     _v1 = vfmaq_f32(_bias1, _v1, _scale1);
                    //     vst1_f16(ptr0, vcvt_f16_f32(_v0));
                    //     vst1_f16(ptr1, vcvt_f16_f32(_v1));

                    //     intptr += 8;
                    //     ptr0 += 4;
                    //     ptr1 += 4;
                    // }
                    vl = 4;
                    vfloat32m2_t _scale0 = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + i * 8, vl);
                    vfloat32m2_t _scale1 = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + i * 8 + 4, vl);
                    vfloat32m2_t _bias0 = bias_data_size == 1 ? vfmv_v_f_f32m2(bias_data[0], vl) : vle32_v_f32m2((const float*)bias_data + i * 8, vl);
                    vfloat32m2_t _bias1 = bias_data_size == 1 ? vfmv_v_f_f32m2(bias_data[0], vl) : vle32_v_f32m2((const float*)bias_data + i * 8 + 4, vl);

                    for (int j = 0; j < w; j++)
                    {
                        vfloat32m2_t _v0 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        vfloat32m2_t _v1 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr + 4, vl), vl);
                        _v0 = vfmadd_vv_f32m2(_v0, _scale0, _bias0, vl);
                        _v1 = vfmadd_vv_f32m2(_v1, _scale1, _bias1, vl);
                        // vse32_v_f32m1(ptr0, _v0, vl);
                        // vse32_v_f32m1(ptr1, _v1, vl);
                        vse16_v_f16m1(ptr0, vfncvt_f_f_w_f16m1(_v0, vl), vl);
                        vse16_v_f16m1(ptr1, vfncvt_f_f_w_f16m1(_v1, vl), vl);
                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
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
            int outc = channels * 2;

            top_blob.create(w, h, outc, (size_t)8u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    __fp16* ptr0 = top_blob.channel(q * 2);
                    __fp16* ptr1 = top_blob.channel(q * 2 + 1);

                    vl = 4;
                    vfloat32m2_t _scale0 = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + q * 8, vl);
                    vfloat32m2_t _scale1 = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + q * 8 + 4, vl);

                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m2_t _v0 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        vfloat32m2_t _v1 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr + 4, vl), vl);
                        _v0 = vfmul_vv_f32m2(_v0, _scale0, vl);
                        _v1 = vfmul_vv_f32m2(_v1, _scale1, vl);
                        // vse32_v_f32m1(ptr0, _v0, vl);
                        // vse32_v_f32m1(ptr1, _v1, vl);
                        vse16_v_f16m1(ptr0, vfncvt_f_f_w_f16m1(_v0, vl), vl);
                        vse16_v_f16m1(ptr1, vfncvt_f_f_w_f16m1(_v1, vl), vl);
                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                    // float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8);
                    // float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8 + 4);

                    // for (int i = 0; i < size; i++)
                    // {
                    //     float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                    //     _v0 = vmulq_f32(_v0, _scale0);
                    //     _v1 = vmulq_f32(_v1, _scale1);
                    //     vst1_f16(ptr0, vcvt_f16_f32(_v0));
                    //     vst1_f16(ptr1, vcvt_f16_f32(_v1));

                    //     intptr += 8;
                    //     ptr0 += 4;
                    //     ptr1 += 4;
                    // }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    __fp16* ptr0 = top_blob.channel(q * 2);
                    __fp16* ptr1 = top_blob.channel(q * 2 + 1);

                    // float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8);
                    // float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8 + 4);
                    // float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 8);
                    // float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 8 + 4);

                    // for (int i = 0; i < size; i++)
                    // {
                    //     float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                    //     _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                    //     _v1 = vfmaq_f32(_bias1, _v1, _scale1);
                    //     vst1_f16(ptr0, vcvt_f16_f32(_v0));
                    //     vst1_f16(ptr1, vcvt_f16_f32(_v1));

                    //     intptr += 8;
                    //     ptr0 += 4;
                    //     ptr1 += 4;
                    // }
                    vl = 4;
                    vfloat32m2_t _scale0 = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + q * 8, vl);
                    vfloat32m2_t _scale1 = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + q * 8 + 4, vl);
                    vfloat32m2_t _bias0 = bias_data_size == 1 ? vfmv_v_f_f32m2(bias_data[0], vl) : vle32_v_f32m2((const float*)bias_data + q * 8, vl);
                    vfloat32m2_t _bias1 = bias_data_size == 1 ? vfmv_v_f_f32m2(bias_data[0], vl) : vle32_v_f32m2((const float*)bias_data + q * 8 + 4, vl);

                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m2_t _v0 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        vfloat32m2_t _v1 = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr + 4, vl), vl);
                        _v0 = vfmadd_vv_f32m2(_v0, _scale0, _bias0, vl);
                        _v1 = vfmadd_vv_f32m2(_v1, _scale1, _bias1, vl);
                        // vse32_v_f32m1(ptr0, _v0, vl);
                        // vse32_v_f32m1(ptr1, _v1, vl);
                        vse16_v_f16m1(ptr0, vfncvt_f_f_w_f16m1(_v0, vl), vl);
                        vse16_v_f16m1(ptr1, vfncvt_f_f_w_f16m1(_v1, vl), vl);
                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
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

            top_blob.create(w, (size_t)8u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                // float32x4_t _scale = vdupq_n_f32(scale_data[0]);

                if (bias_data_size == 0)
                {
                    // #pragma omp parallel for num_threads(opt.num_threads)
                    // for (int i = 0; i < w; i++)
                    // {
                    //     const int* intptr = (const int*)bottom_blob + i * 4;
                    //     __fp16* ptr = (__fp16*)top_blob + i * 4;

                    //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     _v = vmulq_f32(_v, _scale);
                    //     vst1_f16(ptr, vcvt_f16_f32(_v));
                    // }
                    int n = w * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        __fp16* ptr = (__fp16*)top_blob + offset;
                        vfloat32m8_t _scale = vfmv_v_f_f32m8(scale_data[0], vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmul_vv_f32m8(_v, _scale, vl);
                        // vse32_v_f32m8(ptr, _v, vl);
                        vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_v, vl), vl);
                        offset += vl;
                        n -= vl;
                    }
                }
                else if (bias_data_size == 1)
                {
                    // float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    // #pragma omp parallel for num_threads(opt.num_threads)
                    // for (int i = 0; i < w; i++)
                    // {
                    //     const int* intptr = (const int*)bottom_blob + i * 4;
                    //     __fp16* ptr = (__fp16*)top_blob + i * 4;

                    //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     _v = vfmaq_f32(_bias, _v, _scale);
                    //     vst1_f16(ptr, vcvt_f16_f32(_v));
                    // }
                    int n = w * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        __fp16* ptr = (__fp16*)top_blob + offset;
                        vfloat32m8_t _scale = vfmv_v_f_f32m8(scale_data[0], vl);
                        vfloat32m8_t _bias = vfmv_v_f_f32m8(bias_data[0], vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                        // vse32_v_f32m8(ptr, _v, vl);
                        vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_v, vl), vl);
                        offset += vl;
                        n -= vl;
                    }
                }
                else
                {
                    // #pragma omp parallel for num_threads(opt.num_threads)
                    // for (int i = 0; i < w; i++)
                    // {
                    //     const int* intptr = (const int*)bottom_blob + i * 4;
                    //     __fp16* ptr = (__fp16*)top_blob + i * 4;

                    //     float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                    //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     _v = vfmaq_f32(_bias, _v, _scale);
                    //     vst1_f16(ptr, vcvt_f16_f32(_v));
                    // }
                    int n = w * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        __fp16* ptr = (__fp16*)top_blob + offset;
                        vfloat32m8_t _scale = vfmv_v_f_f32m8(scale_data[0], vl);
                        vfloat32m8_t _bias = vle32_v_f32m8((const float*)bias_data + offset, vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                        // vse32_v_f32m8(ptr, _v, vl);
                        vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_v, vl), vl);
                        offset += vl;
                        n -= vl;
                    }
                }
            }
            else
            {
                if (bias_data_size == 0)
                {
                    // #pragma omp parallel for num_threads(opt.num_threads)
                    // for (int i = 0; i < w; i++)
                    // {
                    //     const int* intptr = (const int*)bottom_blob + i * 4;
                    //     __fp16* ptr = (__fp16*)top_blob + i * 4;

                    //     float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                    //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     _v = vmulq_f32(_v, _scale);
                    //     vst1_f16(ptr, vcvt_f16_f32(_v));
                    // }
                    int n = w * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        __fp16* ptr = (__fp16*)top_blob + offset;
                        vfloat32m8_t _scale = vle32_v_f32m8((const float*)scale_data + offset, vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmul_vv_f32m8(_v, _scale, vl);
                        // vse32_v_f32m8(ptr, _v, vl);
                        vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_v, vl), vl);
                        offset += vl;
                        n -= vl;
                    }
                }
                else if (bias_data_size == 1)
                {
                    // float32x4_t _bias = vdupq_n_f32(bias_data[0]);

                    // #pragma omp parallel for num_threads(opt.num_threads)
                    // for (int i = 0; i < w; i++)
                    // {
                    //     const int* intptr = (const int*)bottom_blob + i * 4;
                    //     __fp16* ptr = (__fp16*)top_blob + i * 4;

                    //     float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                    //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     _v = vfmaq_f32(_bias, _v, _scale);
                    //     vst1_f16(ptr, vcvt_f16_f32(_v));
                    // }
                    int n = w * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        __fp16* ptr = (__fp16*)top_blob + offset;
                        vfloat32m8_t _scale = vle32_v_f32m8((const float*)scale_data + offset, vl);
                        vfloat32m8_t _bias = vfmv_v_f_f32m8(bias_data[0], vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                        // vse32_v_f32m8(ptr, _v, vl);
                        vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_v, vl), vl);
                        offset += vl;
                        n -= vl;
                    }
                }
                else
                {
                    // #pragma omp parallel for num_threads(opt.num_threads)
                    // for (int i = 0; i < w; i++)
                    // {
                    //     const int* intptr = (const int*)bottom_blob + i * 4;
                    //     __fp16* ptr = (__fp16*)top_blob + i * 4;

                    //     float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                    //     float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                    //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     _v = vfmaq_f32(_bias, _v, _scale);
                    //     vst1_f16(ptr, vcvt_f16_f32(_v));
                    // }
                    int n = w * 4;
                    int offset = 0;
                    while (n > 0)
                    {
                        vl = vsetvl_e32m8(n);
                        const int* intptr = (const int*)bottom_blob + offset;
                        __fp16* ptr = (__fp16*)top_blob + offset;
                        vfloat32m8_t _scale = vle32_v_f32m8((const float*)scale_data + offset, vl);
                        vfloat32m8_t _bias = vle32_v_f32m8((const float*)bias_data + offset, vl);
                        vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr, vl), vl);
                        _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                        // vse32_v_f32m8(ptr, _v, vl);
                        vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_v, vl), vl);
                        offset += vl;
                        n -= vl;
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;

            top_blob.create(w, h, (size_t)8u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    __fp16* ptr = top_blob.row<__fp16>(i);

                    // float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 4);
                    vl = 4;
                    vfloat32m2_t _scale = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + i * 4, vl);
                    for (int j = 0; j < w; j++)
                    {
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmul_vv_f32m2(_v, _scale, vl);
                        // vse32_v_f32m1(ptr, _v, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);
                        intptr += 4;
                        ptr += 4;
                    }

                    // for (int j = 0; j < w; j++)
                    // {
                    //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     _v = vmulq_f32(_v, _scale);
                    //     vst1_f16(ptr, vcvt_f16_f32(_v));

                    //     intptr += 4;
                    //     ptr += 4;
                    // }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    __fp16* ptr = top_blob.row<__fp16>(i);

                    // float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 4);
                    // float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 4);

                    // for (int j = 0; j < w; j++)
                    // {
                    //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     _v = vfmaq_f32(_bias, _v, _scale);
                    //     vst1_f16(ptr, vcvt_f16_f32(_v));

                    //     intptr += 4;
                    //     ptr += 4;
                    // }
                    vl = 4;
                    vfloat32m2_t _scale = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + i * 4, vl);
                    vfloat32m2_t _bias = bias_data_size == 1 ? vfmv_v_f_f32m2(bias_data[0], vl) : vle32_v_f32m2((const float*)bias_data + i * 4, vl);

                    for (int j = 0; j < w; j++)
                    {
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmadd_vv_f32m2(_v, _scale, _bias, vl);
                        // vse32_v_f32m1(ptr, _v, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);
                        intptr += 4;
                        ptr += 4;
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

            top_blob.create(w, h, channels, (size_t)8u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    __fp16* ptr = top_blob.channel(q);

                    // float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 4);

                    // for (int i = 0; i < size; i++)
                    // {
                    //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     _v = vmulq_f32(_v, _scale);
                    //     vst1_f16(ptr, vcvt_f16_f32(_v));

                    //     intptr += 4;
                    //     ptr += 4;
                    // }
                    vl = 4;
                    vfloat32m2_t _scale = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + q * 4, vl);
                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmul_vv_f32m2(_v, _scale, vl);
                        // vse32_v_f32m1(ptr, _v, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);
                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    __fp16* ptr = top_blob.channel(q);

                    // float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 4);
                    // float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 4);

                    // for (int i = 0; i < size; i++)
                    // {
                    //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    //     _v = vfmaq_f32(_bias, _v, _scale);
                    //     vst1_f16(ptr, vcvt_f16_f32(_v));

                    //     intptr += 4;
                    //     ptr += 4;
                    // }
                    vl = 4;
                    vfloat32m2_t _scale = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + q * 4, vl);
                    vfloat32m2_t _bias = bias_data_size == 1 ? vfmv_v_f_f32m2(bias_data[0], vl) : vle32_v_f32m2((const float*)bias_data + q * 4, vl);

                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmadd_vv_f32m2(_v, _scale, _bias, vl);
                        // vse32_v_f32m1(ptr, _v, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);
                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
        }

        return 0;
    }

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)2u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int* intptr = bottom_blob;
        __fp16* ptr = top_blob;

        if (scale_data_size == 1)
        {
            const float scale = scale_data[0];

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = (__fp16)(intptr[i] * scale);
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = (__fp16)(intptr[i] * scale + bias);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = (__fp16)(intptr[i] * scale + bias_data[i]);
                }
            }
        }
        else
        {
            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = (__fp16)(intptr[i] * scale_data[i]);
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = (__fp16)(intptr[i] * scale_data[i] + bias);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = (__fp16)(intptr[i] * scale_data[i] + bias_data[i]);
                }
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)2u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                __fp16* ptr = top_blob.row<__fp16>(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

                int j = 0;

                int n = w;
                int offset = 0;
                while (n > 0)
                {
                    vl = vsetvl_e32m8(n);
                    vfloat32m8_t _scale = vfmv_v_f_f32m8(scale, vl);
                    vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr + offset, vl), vl);
                    _v = vfmul_vv_f32m8(_v, _scale, vl);
                    // vse32_v_f32m8(ptr + offset, _v, vl);
                    vse16_v_f16m4(ptr + offset, vfncvt_f_f_w_f16m4(_v, vl), vl);
                    offset += vl;
                    n -= vl;
                }
                // float32x4_t _scale = vdupq_n_f32(scale);
                // for (; j + 3 < w; j += 4)
                // {
                //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                //     _v = vmulq_f32(_v, _scale);
                //     vst1_f16(ptr, vcvt_f16_f32(_v));

                //     intptr += 4;
                //     ptr += 4;
                // }
                // for (; j < w; j++)
                // {
                //     *ptr++ = (__fp16)(*intptr++ * scale);
                // }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                __fp16* ptr = top_blob.row<__fp16>(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[i];

                int j = 0;
                // float32x4_t _scale = vdupq_n_f32(scale);
                // float32x4_t _bias = vdupq_n_f32(bias);
                // for (; j + 3 < w; j += 4)
                // {
                //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                //     _v = vfmaq_f32(_bias, _v, _scale);
                //     vst1_f16(ptr, vcvt_f16_f32(_v));

                //     intptr += 4;
                //     ptr += 4;
                // }
                // for (; j < w; j++)
                // {
                //     *ptr++ = (__fp16)(*intptr++ * scale + bias);
                // }
                int n = w;
                int offset = 0;
                while (n > 0)
                {
                    vl = vsetvl_e32m8(n);
                    vfloat32m8_t _scale = vfmv_v_f_f32m8(scale, vl);
                    vfloat32m8_t _bias = vfmv_v_f_f32m8(bias, vl);
                    vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr + offset, vl), vl);
                    _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                    // vse32_v_f32m8(ptr + offset, _v, vl);
                    vse16_v_f16m4(ptr + offset, vfncvt_f_f_w_f16m4(_v, vl), vl);
                    offset += vl;
                    n -= vl;
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

        top_blob.create(w, h, channels, (size_t)2u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                __fp16* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

                int i = 0;
                // float32x4_t _scale = vdupq_n_f32(scale);
                // for (; i + 3 < size; i += 4)
                // {
                //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                //     _v = vmulq_f32(_v, _scale);
                //     vst1_f16(ptr, vcvt_f16_f32(_v));

                //     intptr += 4;
                //     ptr += 4;
                // }
                // for (; i < size; i++)
                // {
                //     *ptr++ = (__fp16)(*intptr++ * scale);
                // }
                int n = size;
                int offset = 0;

                while (n > 0)
                {
                    vl = vsetvl_e32m8(n);
                    vfloat32m8_t _scale = vfmv_v_f_f32m8(scale, vl);
                    vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr + offset, vl), vl);
                    _v = vfmul_vv_f32m8(_v, _scale, vl);
                    // vse32_v_f32m8(ptr + offset, _v, vl);
                    vse16_v_f16m4(ptr + offset, vfncvt_f_f_w_f16m4(_v, vl), vl);
                    offset += vl;
                    n -= vl;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                __fp16* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[q];

                int i = 0;
                // float32x4_t _scale = vdupq_n_f32(scale);
                // float32x4_t _bias = vdupq_n_f32(bias);
                // for (; i + 3 < size; i += 4)
                // {
                //     float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                //     _v = vfmaq_f32(_bias, _v, _scale);
                //     vst1_f16(ptr, vcvt_f16_f32(_v));

                //     intptr += 4;
                //     ptr += 4;
                // }
                // for (; i < size; i++)
                // {
                //     *ptr++ = (__fp16)(*intptr++ * scale + bias);
                // }
                int n = size;
                int offset = 0;

                while (n > 0)
                {
                    vl = vsetvl_e32m8(n);
                    vfloat32m8_t _scale = vfmv_v_f_f32m8(scale, vl);
                    vfloat32m8_t _bias = vfmv_v_f_f32m8(bias, vl);
                    vfloat32m8_t _v = vfcvt_f_x_v_f32m8(vle32_v_i32m8(intptr + offset, vl), vl);
                    _v = vfmadd_vv_f32m8(_scale, _v, _bias, vl);
                    // vse32_v_f32m8(ptr + offset, _v, vl);
                    vse16_v_f16m4(ptr + offset, vfncvt_f_f_w_f16m4(_v, vl), vl);
                    offset += vl;
                    n -= vl;
                }
            }
        }
    }

    return 0;
}

int Dequantize_riscv::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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

            top_blob.create(w, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                // float32x4_t _scale = vdupq_n_f32(scale_data[0]);
                vfloat32m2_t _scale = vfmv_v_f_f32m2(scale_data[0], vl);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        __fp16* ptr = (__fp16*)top_blob + i * 8;

                        // float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        // float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        // _v0 = vmulq_f32(_v0, _scale);
                        // _v1 = vmulq_f32(_v1, _scale);
                        // vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(_v0), vcvt_f16_f32(_v1)));
                        vl = 8;
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmul_vv_f32m2(_v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);
                    }
                }
                else if (bias_data_size == 1)
                {
                    // float32x4_t _bias = vdupq_n_f32(bias_data[0]);
                    vfloat32m2_t _bias = vfmv_v_f_f32m2(bias_data[0], vl);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        __fp16* ptr = (__fp16*)top_blob + i * 8;

                        // float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        // float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        // _v0 = vfmaq_f32(_bias, _v0, _scale);
                        // _v1 = vfmaq_f32(_bias, _v1, _scale);
                        // vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(_v0), vcvt_f16_f32(_v1)));
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmacc_vv_f32m2(_bias, _v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        __fp16* ptr = (__fp16*)top_blob + i * 8;

                        // float32x4_t _bias0 = vld1q_f32((const float*)bias_data + i * 8);
                        // float32x4_t _bias1 = vld1q_f32((const float*)bias_data + i * 8 + 4);
                        // float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        // float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        // _v0 = vfmaq_f32(_bias0, _v0, _scale);
                        // _v1 = vfmaq_f32(_bias1, _v1, _scale);
                        // vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(_v0), vcvt_f16_f32(_v1)));
                        vfloat32m2_t _bias = vle32_v_f32m2((const float*)bias_data + i * 8, vl);
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmacc_vv_f32m2(_bias, _v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);
                    }
                }
            }
            else
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        __fp16* ptr = (__fp16*)top_blob + i * 8;

                        // float32x4_t _scale0 = vld1q_f32((const float*)scale_data + i * 8);
                        // float32x4_t _scale1 = vld1q_f32((const float*)scale_data + i * 8 + 4);
                        // float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        // float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        // _v0 = vmulq_f32(_v0, _scale0);
                        // _v1 = vmulq_f32(_v1, _scale1);
                        // vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(_v0), vcvt_f16_f32(_v1)));
                        vfloat32m2_t _scale = vle32_v_f32m2((const float*)scale_data + i * 8, vl);
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmul_vv_f32m2(_v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);
                    }
                }
                else if (bias_data_size == 1)
                {
                    // float32x4_t _bias = vdupq_n_f32(bias_data[0]);
                    vfloat32m2_t _bias = vfmv_v_f_f32m2(bias_data[0], vl);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        __fp16* ptr = (__fp16*)top_blob + i * 8;

                        // float32x4_t _scale0 = vld1q_f32((const float*)scale_data + i * 8);
                        // float32x4_t _scale1 = vld1q_f32((const float*)scale_data + i * 8 + 4);
                        // float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        // float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        // _v0 = vfmaq_f32(_bias, _v0, _scale0);
                        // _v1 = vfmaq_f32(_bias, _v1, _scale1);
                        // vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(_v0), vcvt_f16_f32(_v1)));
                        vfloat32m2_t _scale = vle32_v_f32m2((const float*)scale_data + i * 8, vl);
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmacc_vv_f32m2(_bias, _v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        __fp16* ptr = (__fp16*)top_blob + i * 8;

                        // float32x4_t _scale0 = vld1q_f32((const float*)scale_data + i * 8);
                        // float32x4_t _scale1 = vld1q_f32((const float*)scale_data + i * 8 + 4);
                        // float32x4_t _bias0 = vld1q_f32((const float*)bias_data + i * 8);
                        // float32x4_t _bias1 = vld1q_f32((const float*)bias_data + i * 8 + 4);
                        // float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        // float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        // _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                        // _v1 = vfmaq_f32(_bias1, _v1, _scale1);
                        // vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(_v0), vcvt_f16_f32(_v1)));
                        vfloat32m2_t _scale = vle32_v_f32m2((const float*)scale_data + i * 8, vl);
                        vfloat32m2_t _bias = vle32_v_f32m2((const float*)bias_data + i * 8, vl);
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmacc_vv_f32m2(_bias, _v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;

            top_blob.create(w, h, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    __fp16* ptr = top_blob.row<__fp16>(i);

                    // float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8);
                    // float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8 + 4);
                    vfloat32m2_t _scale0 = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + i * 8, vl);

                    for (int j = 0; j < w; j++)
                    {
                        // float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        // float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        // _v0 = vmulq_f32(_v0, _scale0);
                        // _v1 = vmulq_f32(_v1, _scale1);
                        // vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(_v0), vcvt_f16_f32(_v1)));
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmul_vv_f32m2(_v, _scale0, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);

                        intptr += 8;
                        ptr += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    __fp16* ptr = top_blob.row<__fp16>(i);

                    // float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8);
                    // float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 8 + 4);
                    // float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8);
                    // float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 8 + 4);
                    vfloat32m2_t _scale = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + i * 8, vl);
                    vfloat32m2_t _bias = bias_data_size == 1 ? vfmv_v_f_f32m2(bias_data[0], vl) : vle32_v_f32m2((const float*)bias_data + i * 8, vl);

                    for (int j = 0; j < w; j++)
                    {
                        // float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        // float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        // _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                        // _v1 = vfmaq_f32(_bias1, _v1, _scale1);
                        // vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(_v0), vcvt_f16_f32(_v1)));
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmacc_vv_f32m2(_bias, _v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);

                        intptr += 8;
                        ptr += 8;
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

            top_blob.create(w, h, channels, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    __fp16* ptr = top_blob.channel(q);

                    // float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8);
                    // float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8 + 4);
                    vfloat32m2_t _scale0 = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + q * 8, vl);

                    for (int i = 0; i < size; i++)
                    {
                        // float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        // float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        // _v0 = vmulq_f32(_v0, _scale0);
                        // _v1 = vmulq_f32(_v1, _scale1);
                        // vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(_v0), vcvt_f16_f32(_v1)));

                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmul_vv_f32m2(_v, _scale0, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);

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
                    __fp16* ptr = top_blob.channel(q);

                    // float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8);
                    // float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 8 + 4);
                    // float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 8);
                    // float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 8 + 4);
                    vfloat32m2_t _scale = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + q * 8, vl);
                    vfloat32m2_t _bias = bias_data_size == 1 ? vfmv_v_f_f32m2(bias_data[0], vl) : vle32_v_f32m2((const float*)bias_data + q * 8, vl);

                    for (int i = 0; i < size; i++)
                    {
                        // float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        // float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        // _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                        // _v1 = vfmaq_f32(_bias1, _v1, _scale1);
                        // vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(_v0), vcvt_f16_f32(_v1)));
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmacc_vv_f32m2(_bias, _v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);

                        intptr += 8;
                        ptr += 8;
                    }
                }
            }
        }

        return 0;
    }

    if (elempack == 4)
    {
        vl = 4;
        if (dims == 1)
        {
            int w = bottom_blob.w;

            top_blob.create(w, (size_t)8u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                // float32x4_t _scale = vdupq_n_f32(scale_data[0]);
                vfloat32m2_t _scale = vfmv_v_f_f32m2(scale_data[0], vl);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        // float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        // _v = vmulq_f32(_v, _scale);
                        // vst1_f16(ptr, vcvt_f16_f32(_v));
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmul_vv_f32m2(_v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);
                    }
                }
                else if (bias_data_size == 1)
                {
                    // float32x4_t _bias = vdupq_n_f32(bias_data[0]);
                    vfloat32m2_t _bias = vfmv_v_f_f32m2(bias_data[0], vl);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        // float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        // _v = vfmaq_f32(_bias, _v, _scale);
                        // vst1_f16(ptr, vcvt_f16_f32(_v));
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmacc_vv_f32m2(_bias, _v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        // float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                        // float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        // _v = vfmaq_f32(_bias, _v, _scale);
                        // vst1_f16(ptr, vcvt_f16_f32(_v));
                        vfloat32m2_t _bias = vle32_v_f32m2((const float*)bias_data + i * 4, vl);
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmacc_vv_f32m2(_bias, _v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);
                    }
                }
            }
            else
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        // float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        // float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        // _v = vmulq_f32(_v, _scale);
                        // vst1_f16(ptr, vcvt_f16_f32(_v));
                        vfloat32m2_t _scale = vle32_v_f32m2((const float*)scale_data + i * 4, vl);
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmul_vv_f32m2(_v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);
                    }
                }
                else if (bias_data_size == 1)
                {
                    // float32x4_t _bias = vdupq_n_f32(bias_data[0]);
                    vfloat32m2_t _bias = vfmv_v_f_f32m2(bias_data[0], vl);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        // float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        // float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        // _v = vfmaq_f32(_bias, _v, _scale);
                        // vst1_f16(ptr, vcvt_f16_f32(_v));
                        vfloat32m2_t _scale = vle32_v_f32m2((const float*)scale_data + i * 4, vl);
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmacc_vv_f32m2(_bias, _v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        __fp16* ptr = (__fp16*)top_blob + i * 4;

                        // float32x4_t _scale = vld1q_f32((const float*)scale_data + i * 4);
                        // float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);
                        // float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        // _v = vfmaq_f32(_bias, _v, _scale);
                        // vst1_f16(ptr, vcvt_f16_f32(_v));
                        vfloat32m2_t _scale = vle32_v_f32m2((const float*)scale_data + i * 4, vl);
                        vfloat32m2_t _bias = vle32_v_f32m2((const float*)bias_data + i * 4, vl);
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmacc_vv_f32m2(_bias, _v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;

            top_blob.create(w, h, (size_t)8u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    __fp16* ptr = top_blob.row<__fp16>(i);

                    // float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 4);
                    vfloat32m2_t _scale = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + i * 4, vl);

                    for (int j = 0; j < w; j++)
                    {
                        // float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        // _v = vmulq_f32(_v, _scale);
                        // vst1_f16(ptr, vcvt_f16_f32(_v));
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmul_vv_f32m2(_v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    __fp16* ptr = top_blob.row<__fp16>(i);

                    // float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + i * 4);
                    // float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + i * 4);
                    vfloat32m2_t _scale = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + i * 4, vl);
                    vfloat32m2_t _bias = bias_data_size == 1 ? vfmv_v_f_f32m2(bias_data[0], vl) : vle32_v_f32m2((const float*)bias_data + i * 4, vl);

                    for (int j = 0; j < w; j++)
                    {
                        // float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        // _v = vfmaq_f32(_bias, _v, _scale);
                        // vst1_f16(ptr, vcvt_f16_f32(_v));
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmacc_vv_f32m2(_bias, _v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);

                        intptr += 4;
                        ptr += 4;
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

            top_blob.create(w, h, channels, (size_t)8u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    __fp16* ptr = top_blob.channel(q);

                    // float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 4);
                    vfloat32m2_t _scale = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + q * 4, vl);

                    for (int i = 0; i < size; i++)
                    {
                        // float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        // _v = vmulq_f32(_v, _scale);
                        // vst1_f16(ptr, vcvt_f16_f32(_v));
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmul_vv_f32m2(_v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    __fp16* ptr = top_blob.channel(q);

                    // float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data[0]) : vld1q_f32((const float*)scale_data + q * 4);
                    // float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data[0]) : vld1q_f32((const float*)bias_data + q * 4);
                    vfloat32m2_t _scale = scale_data_size == 1 ? vfmv_v_f_f32m2(scale_data[0], vl) : vle32_v_f32m2((const float*)scale_data + q * 4, vl);
                    vfloat32m2_t _bias = bias_data_size == 1 ? vfmv_v_f_f32m2(bias_data[0], vl) : vle32_v_f32m2((const float*)bias_data + q * 4, vl);

                    for (int i = 0; i < size; i++)
                    {
                        // float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        // _v = vfmaq_f32(_bias, _v, _scale);
                        // vst1_f16(ptr, vcvt_f16_f32(_v));
                        vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                        _v = vfmacc_vv_f32m2(_bias, _v, _scale, vl);
                        vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
        }

        return 0;
    }

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)2u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int* intptr = bottom_blob;
        __fp16* ptr = top_blob;

        if (scale_data_size == 1)
        {
            const float scale = scale_data[0];

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = (__fp16)(intptr[i] * scale);
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = (__fp16)(intptr[i] * scale + bias);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = (__fp16)(intptr[i] * scale + bias_data[i]);
                }
            }
        }
        else
        {
            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = (__fp16)(intptr[i] * scale_data[i]);
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = (__fp16)(intptr[i] * scale_data[i] + bias);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = (__fp16)(intptr[i] * scale_data[i] + bias_data[i]);
                }
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)2u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                __fp16* ptr = top_blob.row<__fp16>(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

                int j = 0;
                vl = 4;
                // float32x4_t _scale = vdupq_n_f32(scale);
                vfloat32m2_t _scale = vfmv_v_f_f32m2(scale, vl);
                for (; j + 3 < w; j += 4)
                {
                    // float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    // _v = vmulq_f32(_v, _scale);
                    // vst1_f16(ptr, vcvt_f16_f32(_v));
                    vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                    _v = vfmul_vv_f32m2(_v, _scale, vl);
                    vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);

                    intptr += 4;
                    ptr += 4;
                }
                for (; j < w; j++)
                {
                    *ptr++ = (__fp16)(*intptr++ * scale);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                __fp16* ptr = top_blob.row<__fp16>(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[i];

                int j = 0;
                vl = 4;
                // float32x4_t _scale = vdupq_n_f32(scale);
                // float32x4_t _bias = vdupq_n_f32(bias);
                vfloat32m2_t _scale = vfmv_v_f_f32m2(scale, vl);
                vfloat32m2_t _bias = vfmv_v_f_f32m2(bias, vl);
                for (; j + 3 < w; j += 4)
                {
                    // float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    // _v = vfmaq_f32(_bias, _v, _scale);
                    // vst1_f16(ptr, vcvt_f16_f32(_v));
                    vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                    _v = vfmacc_vv_f32m2(_bias, _v, _scale, vl);
                    vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);

                    intptr += 4;
                    ptr += 4;
                }
                for (; j < w; j++)
                {
                    *ptr++ = (__fp16)(*intptr++ * scale + bias);
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

        top_blob.create(w, h, channels, (size_t)2u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                __fp16* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

                int i = 0;
                vl = 4;
                vfloat32m2_t _scale = vfmv_v_f_f32m2(scale, vl);
                // float32x4_t _scale = vdupq_n_f32(scale);
                for (; i + 3 < size; i += 4)
                {
                    // float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    // _v = vmulq_f32(_v, _scale);
                    // vst1_f16(ptr, vcvt_f16_f32(_v));
                    vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                    _v = vfmul_vv_f32m2(_v, _scale, vl);
                    vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);

                    intptr += 4;
                    ptr += 4;
                }
                for (; i < size; i++)
                {
                    *ptr++ = (__fp16)(*intptr++ * scale);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                __fp16* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[q];

                int i = 0;
                // float32x4_t _scale = vdupq_n_f32(scale);
                // float32x4_t _bias = vdupq_n_f32(bias);
                vl = 4;
                vfloat32m2_t _scale = vfmv_v_f_f32m2(scale, vl);
                vfloat32m2_t _bias = vfmv_v_f_f32m2(bias, vl);
                for (; i + 3 < size; i += 4)
                {
                    // float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                    // _v = vfmaq_f32(_bias, _v, _scale);
                    // vst1_f16(ptr, vcvt_f16_f32(_v));
                    vfloat32m2_t _v = vfcvt_f_x_v_f32m2(vle32_v_i32m2(intptr, vl), vl);
                    _v = vfmacc_vv_f32m2(_bias, _v, _scale, vl);
                    vse16_v_f16m1(ptr, vfncvt_f_f_w_f16m1(_v, vl), vl);

                    intptr += 4;
                    ptr += 4;
                }
                for (; i < size; i++)
                {
                    *ptr++ = (__fp16)(*intptr++ * scale + bias);
                }
            }
        }
    }

    return 0;
}

#endif // __riscv_vector && __riscv_zfh

} // namespace ncnn
