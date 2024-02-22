// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "eltwise_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#include "rvv_mathfun_fp16s.h"
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

Eltwise_riscv::Eltwise_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif
}

int Eltwise_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int elembits = bottom_blobs[0].elembits();

    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blobs, top_blobs, opt);
        else
            return forward_fp16s(bottom_blobs, top_blobs, opt);
    }

    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h * d * elempack;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (op_type == Operation_PROD)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = vsetvl_e32m8(n);
                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                vfloat32m8_t _outp = vfmul_vv_f32m8(_p, _p1, vl);
                vse32_v_f32m8(outptr, _outp, vl);
                n -= vl;
                ptr += vl;
                ptr1 += vl;
                outptr += vl;
            }
#else
            for (int i = 0; i < size; i++)
            {
                *outptr = *ptr * *ptr1;
                ptr += 1;
                ptr1 += 1;
                outptr += 1;
            }
#endif
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(outptr, vl);
                    vfloat32m8_t _p1 = vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _outp = vfmul_vv_f32m8(_p, _p1, vl);
                    vse32_v_f32m8(outptr, _outp, vl);
                    n -= vl;
                    ptr += vl;
                    outptr += vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    *outptr *= *ptr;
                    ptr += 1;
                    outptr += 1;
                }
#endif
            }
        }
    }
    if (op_type == Operation_SUM)
    {
        if (coeffs.w == 0)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                    vfloat32m8_t _outp = vfadd_vv_f32m8(_p, _p1, vl);
                    vse32_v_f32m8(outptr, _outp, vl);
                    n -= vl;
                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    *outptr = *ptr + *ptr1;
                    ptr += 1;
                    ptr1 += 1;
                    outptr += 1;
                }
#endif
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
#pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

#if __riscv_vector
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = vsetvl_e32m8(n);
                        vfloat32m8_t _p = vle32_v_f32m8(outptr, vl);
                        vfloat32m8_t _p1 = vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _outp = vfadd_vv_f32m8(_p, _p1, vl);
                        vse32_v_f32m8(outptr, _outp, vl);
                        n -= vl;
                        ptr += vl;
                        outptr += vl;
                    }
#else
                    for (int i = 0; i < size; i++)
                    {
                        *outptr += *ptr;
                        ptr += 1;
                        outptr += 1;
                    }
#endif
                }
            }
        }
        else
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                const float coeff0 = coeffs[0];
                const float coeff1 = coeffs[1];

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                    _p = vfmul_vf_f32m8(_p, coeff0, vl);
                    vfloat32m8_t _outp = vfmacc_vf_f32m8(_p, coeff1, _p1, vl);
                    vse32_v_f32m8(outptr, _outp, vl);
                    n -= vl;
                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    *outptr = *ptr * coeff0 + *ptr1 * coeff1;
                    ptr += 1;
                    ptr1 += 1;
                    outptr += 1;
                }
#endif
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
#pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    const float coeff = coeffs[b];

#if __riscv_vector
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = vsetvl_e32m8(n);
                        vfloat32m8_t _p = vle32_v_f32m8(outptr, vl);
                        vfloat32m8_t _p1 = vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _outp = vfmacc_vf_f32m8(_p, coeff, _p1, vl);
                        vse32_v_f32m8(outptr, _outp, vl);
                        n -= vl;
                        ptr += vl;
                        outptr += vl;
                    }
#else
                    for (int i = 0; i < size; i++)
                    {
                        *outptr += *ptr * coeff;
                        ptr += 1;
                        outptr += 1;
                    }
#endif
                }
            }
        }
    }
    if (op_type == Operation_MAX)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = vsetvl_e32m8(n);
                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _p1 = vle32_v_f32m8(ptr1, vl);
                vfloat32m8_t _outp = vfmax_vv_f32m8(_p, _p1, vl);
                vse32_v_f32m8(outptr, _outp, vl);
                n -= vl;
                ptr += vl;
                ptr1 += vl;
                outptr += vl;
            }
#else
            for (int i = 0; i < size; i++)
            {
                *outptr = std::max(*ptr, *ptr1);
                ptr += 1;
                ptr1 += 1;
                outptr += 1;
            }
#endif
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _p = vle32_v_f32m8(outptr, vl);
                    vfloat32m8_t _p1 = vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _outp = vfmax_vv_f32m8(_p, _p1, vl);
                    vse32_v_f32m8(outptr, _outp, vl);
                    n -= vl;
                    ptr += vl;
                    outptr += vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    *outptr = std::max(*ptr, *outptr);
                    ptr += 1;
                    outptr += 1;
                }
#endif
            }
        }
    }

    return 0;
}

#if __riscv_vector && __riscv_zfh
int Eltwise_riscv::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h * d * elempack;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (bottom_blobs.size() == 2)
    {
        // fast path without fp32 accumulator
        if (op_type == Operation_PROD)
        {
            const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);

                int n = size;
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                    vfloat16m8_t _outp = vfmul_vv_f16m8(_p, _p1, vl);
                    vse16_v_f16m8(outptr, _outp, vl);
                    n -= vl;
                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                }
            }
        }
        if (op_type == Operation_SUM)
        {
            if (coeffs.w == 0)
            {
                const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);
                    const __fp16* ptr1 = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = vsetvl_e16m8(n);
                        vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                        vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                        vfloat16m8_t _outp = vfadd_vv_f16m8(_p, _p1, vl);
                        vse16_v_f16m8(outptr, _outp, vl);
                        n -= vl;
                        ptr += vl;
                        ptr1 += vl;
                        outptr += vl;
                    }
                }
            }
            else
            {
                const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);
                    const __fp16* ptr1 = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    const float coeff0 = coeffs[0];
                    const float coeff1 = coeffs[1];
                    __fp16 _coeff0 = (__fp16)coeff0;
                    __fp16 _coeff1 = (__fp16)coeff1;

                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = vsetvl_e16m8(n);
                        vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                        vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                        _p = vfmul_vf_f16m8(_p, coeff0, vl);
                        vfloat16m8_t _outp = vfmacc_vf_f16m8(_p, coeff1, _p1, vl);
                        vse16_v_f16m8(outptr, _outp, vl);
                        n -= vl;
                        ptr += vl;
                        ptr1 += vl;
                        outptr += vl;
                    }
                }
            }
        }
        if (op_type == Operation_MAX)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);

                int n = size;
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                    vfloat16m8_t _outp = vfmax_vv_f16m8(_p, _p1, vl);
                    vse16_v_f16m8(outptr, _outp, vl);
                    n -= vl;
                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                }
            }
        }

        return 0;
    }

    Mat top_blob_fp32(w, h, d, channels, (size_t)4u * elempack, elempack, opt.workspace_allocator);
    if (top_blob_fp32.empty())
        return -100;

    if (op_type == Operation_PROD)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = bottom_blob.channel(q);
            const __fp16* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob_fp32.channel(q);

            int n = size;
            while (n > 0)
            {
                size_t vl = vsetvl_e16m4(n);
                vfloat16m4_t _p = vle16_v_f16m4(ptr, vl);
                vfloat16m4_t _p1 = vle16_v_f16m4(ptr1, vl);
                vfloat32m8_t _outp = vfwmul_vv_f32m8(_p, _p1, vl);
                vse32_v_f32m8(outptr, _outp, vl);
                n -= vl;
                ptr += vl;
                ptr1 += vl;
                outptr += vl;
            }
        }

        size_t b = 2;
        for (; b < bottom_blobs.size() - 1; b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);

                int n = size;
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m4(n);
                    vfloat32m8_t _p = vle32_v_f32m8(outptr, vl);
                    vfloat16m4_t _p1 = vle16_v_f16m4(ptr, vl);
                    vfloat32m8_t _outp = vfmul_vv_f32m8(_p, vfwcvt_f_f_v_f32m8(_p1, vl), vl);
                    vse32_v_f32m8(outptr, _outp, vl);
                    n -= vl;
                    ptr += vl;
                    outptr += vl;
                }
            }
        }
        for (; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob1.channel(q);
                const float* ptr0 = top_blob_fp32.channel(q);
                __fp16* outptr = top_blob.channel(q);

                int n = size;
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m4(n);
                    vfloat32m8_t _p = vle32_v_f32m8(ptr0, vl);
                    vfloat16m4_t _p1 = vle16_v_f16m4(ptr, vl);
                    vfloat32m8_t _outp = vfmul_vv_f32m8(_p, vfwcvt_f_f_v_f32m8(_p1, vl), vl);
                    vfloat16m4_t _outp16 = vfncvt_f_f_w_f16m4(_outp, vl);
                    vse16_v_f16m4(outptr, _outp16, vl);
                    n -= vl;
                    ptr += vl;
                    ptr0 += vl;
                    outptr += vl;
                }
            }
        }
    }
    if (op_type == Operation_SUM)
    {
        if (coeffs.w == 0)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);

                int n = size;
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m4(n);
                    vfloat16m4_t _p = vle16_v_f16m4(ptr, vl);
                    vfloat16m4_t _p1 = vle16_v_f16m4(ptr1, vl);
                    vfloat32m8_t _outp = vfwadd_vv_f32m8(_p, _p1, vl);
                    vse32_v_f32m8(outptr, _outp, vl);
                    n -= vl;
                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                }
            }

            size_t b = 2;
            for (; b < bottom_blobs.size() - 1; b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
#pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob_fp32.channel(q);

                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = vsetvl_e16m4(n);
                        vfloat32m8_t _p = vle32_v_f32m8(outptr, vl);
                        vfloat16m4_t _p1 = vle16_v_f16m4(ptr, vl);
                        vfloat32m8_t _outp = vfwadd_wv_f32m8(_p, _p1, vl);
                        vse32_v_f32m8(outptr, _outp, vl);
                        n -= vl;
                        ptr += vl;
                        outptr += vl;
                    }
                }
            }
            for (; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
#pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    const float* ptr0 = top_blob_fp32.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = vsetvl_e16m4(n);
                        vfloat32m8_t _p = vle32_v_f32m8(ptr0, vl);
                        vfloat16m4_t _p1 = vle16_v_f16m4(ptr, vl);
                        vfloat32m8_t _outp = vfwadd_wv_f32m8(_p, _p1, vl);
                        vfloat16m4_t _outp16 = vfncvt_f_f_w_f16m4(_outp, vl);
                        vse16_v_f16m4(outptr, _outp16, vl);
                        n -= vl;
                        ptr += vl;
                        ptr0 += vl;
                        outptr += vl;
                    }
                }
            }
        }
        else
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);

                const float coeff0 = coeffs[0];
                const float coeff1 = coeffs[1];

                int n = size;
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m4(n);
                    vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr, vl), vl);
                    vfloat32m8_t _p1 = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr1, vl), vl);
                    _p = vfmul_vf_f32m8(_p, coeff0, vl);
                    vfloat32m8_t _outp = vfmacc_vf_f32m8(_p, coeff1, _p1, vl);
                    vse32_v_f32m8(outptr, _outp, vl);
                    n -= vl;
                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                }
            }

            size_t b = 2;
            for (; b < bottom_blobs.size() - 1; b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
#pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob_fp32.channel(q);

                    const float coeff = coeffs[b];

                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = vsetvl_e32m8(n);
                        vfloat32m8_t _p = vle32_v_f32m8(outptr, vl);
                        vfloat32m8_t _p1 = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr, vl), vl);
                        vfloat32m8_t _outp = vfmacc_vf_f32m8(_p, coeff, _p1, vl);
                        vse32_v_f32m8(outptr, _outp, vl);
                        n -= vl;
                        ptr += vl;
                        outptr += vl;
                    }
                }
            }
            for (; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
#pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    const float* ptr0 = top_blob_fp32.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    const float coeff = coeffs[b];

                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = vsetvl_e32m8(n);
                        vfloat32m8_t _p = vle32_v_f32m8(ptr0, vl);
                        vfloat32m8_t _p1 = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr, vl), vl);
                        vfloat32m8_t _outp = vfmacc_vf_f32m8(_p, coeff, _p1, vl);
                        vfloat16m4_t _outp16 = vfncvt_f_f_w_f16m4(_outp, vl);
                        vse16_v_f16m4(outptr, _outp16, vl);
                        n -= vl;
                        ptr += vl;
                        ptr0 += vl;
                        outptr += vl;
                    }
                }
            }
        }
    }
    if (op_type == Operation_MAX)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = bottom_blob.channel(q);
            const __fp16* ptr1 = bottom_blob1.channel(q);
            __fp16* outptr = top_blob.channel(q);

            int n = size;
            while (n > 0)
            {
                size_t vl = vsetvl_e16m8(n);
                vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                vfloat16m8_t _outp = vfmax_vv_f16m8(_p, _p1, vl);
                vse16_v_f16m8(outptr, _outp, vl);
                n -= vl;
                ptr += vl;
                ptr1 += vl;
                outptr += vl;
            }
        }

        size_t b = 2;
        for (; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);

                int n = size;
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(outptr, vl);
                    vfloat16m8_t _p1 = vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _outp = vfmax_vv_f16m8(_p, _p1, vl);
                    vse16_v_f16m8(outptr, _outp, vl);
                    n -= vl;
                    ptr += vl;
                    outptr += vl;
                }
            }
        }
    }

    return 0;
}

int Eltwise_riscv::forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() == 2)
    {
        // fast path without fp32 accumulator
        return forward_fp16s(bottom_blobs, top_blobs, opt);
    }

    if (op_type == Operation_MAX)
    {
        return forward_fp16s(bottom_blobs, top_blobs, opt);
    }

    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h * d * elempack;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (op_type == Operation_PROD)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = bottom_blob.channel(q);
            const __fp16* ptr1 = bottom_blob1.channel(q);
            __fp16* outptr = top_blob.channel(q);

            int n = size;
            while (n > 0)
            {
                size_t vl = vsetvl_e16m8(n);
                vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                vfloat16m8_t _outp = vfmul_vv_f16m8(_p, _p1, vl);
                vse16_v_f16m8(outptr, _outp, vl);
                n -= vl;
                ptr += vl;
                ptr1 += vl;
                outptr += vl;
            }
        }

        size_t b = 2;
        for (; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);

                int n = size;
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(outptr, vl);
                    vfloat16m8_t _p1 = vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _outp = vfmul_vv_f16m8(_p, _p1, vl);
                    vse16_v_f16m8(outptr, _outp, vl);
                    n -= vl;
                    ptr += vl;
                    outptr += vl;
                }
            }
        }
    }
    if (op_type == Operation_SUM)
    {
        if (coeffs.w == 0)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);

                int n = size;
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                    vfloat16m8_t _outp = vfadd_vv_f16m8(_p, _p1, vl);
                    vse16_v_f16m8(outptr, _outp, vl);
                    n -= vl;
                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                }
            }

            size_t b = 2;
            for (; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
#pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = vsetvl_e16m8(n);
                        vfloat16m8_t _p = vle16_v_f16m8(outptr, vl);
                        vfloat16m8_t _p1 = vle16_v_f16m8(ptr, vl);
                        vfloat16m8_t _outp = vfadd_vv_f16m8(_p, _p1, vl);
                        vse16_v_f16m8(outptr, _outp, vl);
                        n -= vl;
                        ptr += vl;
                        outptr += vl;
                    }
                }
            }
        }
        else
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);

                const __fp16 coeff0 = (__fp16)coeffs[0];
                const __fp16 coeff1 = (__fp16)coeffs[1];

                int n = size;
                while (n > 0)
                {
                    size_t vl = vsetvl_e16m8(n);
                    vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _p1 = vle16_v_f16m8(ptr1, vl);
                    _p = vfmul_vf_f16m8(_p, coeff0, vl);
                    vfloat16m8_t _outp = vfmacc_vf_f16m8(_p, coeff1, _p1, vl);
                    vse16_v_f16m8(outptr, _outp, vl);
                    n -= vl;
                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                }
            }

            size_t b = 2;
            for (; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
#pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    const __fp16 coeff = (__fp16)coeffs[b];

                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = vsetvl_e16m8(n);
                        vfloat16m8_t _p = vle16_v_f16m8(outptr, vl);
                        vfloat16m8_t _p1 = vle16_v_f16m8(ptr, vl);
                        vfloat16m8_t _outp = vfmacc_vf_f16m8(_p, coeff, _p1, vl);
                        vse16_v_f16m8(outptr, _outp, vl);
                        n -= vl;
                        ptr += vl;
                        outptr += vl;
                    }
                }
            }
        }
    }

    return 0;
}
#endif // __riscv_vector && __riscv_zfh

} // namespace ncnn
