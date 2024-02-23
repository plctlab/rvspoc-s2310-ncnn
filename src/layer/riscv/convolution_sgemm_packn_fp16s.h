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

static void im2col_sgemm_packn_fp16sa_rvv(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    // Mat bottom_im2col(size, maxk, inch, 2u * packn, packn, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const __fp16* bias = _bias;

    // permute
    Mat tmp;
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, 2u * packn, packn, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + (size % 4) / 2 + size % 2, 2u * packn, packn, opt.workspace_allocator);
    else if (size >= 2)
        tmp.create(2 * maxk, inch, size / 2 + size % 2, 2u * packn, packn, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 2u * packn, packn, opt.workspace_allocator);
    {
        int remain_size_start = 0;
        int nn_size = size >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            __fp16* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i * packn;
                for (int k = 0; k < maxk; k++)
                {
                    asm volatile(
                        "mv        t3,     %[LEN]  \n\t"
                        "mv        t1,     %[SRC]  \n\t"
                        "mv        t2,     %[TMP]  \n\t"
                        "slli      t3,     t3,     1       \n\t"
                        "vle.v     v0,     (t1)    \n\t"
                        "add       t1,     t1,     t3      \n\t"
                        "vle.v     v1,     (t1)    \n\t"
                        "add       t1,     t1,     t3      \n\t"
                        "vle.v     v2,     (t1)    \n\t"
                        "add       t1,     t1,     t3      \n\t"
                        "vle.v     v3,     (t1)    \n\t"
                        "add       t1,     t1,     t3      \n\t"
                        "vle.v     v4,     (t1)    \n\t"
                        "add       t1,     t1,     t3      \n\t"
                        "vle.v     v5,     (t1)    \n\t"
                        "add       t1,     t1,     t3      \n\t"
                        "vle.v     v6,     (t1)    \n\t"
                        "add       t1,     t1,     t3      \n\t"
                        "vle.v     v7,     (t1)    \n\t"
                        "add       t1,     t1,     t3      \n\t"
                        "vsseg8e.v v0,     (t2)    \n\t"
                        :
                        : [LEN] "r"(packn), [SRC] "r"(img0), [TMP] "r"(tmpptr)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "t1", "t2", "t3");

                    img0 += size * packn;
                    tmpptr += packn * 8;
                }
            }
        }

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
                    asm volatile(
                        "mv        t3,     %[LEN]  \n\t"
                        "mv        t1,     %[SRC]  \n\t"
                        "mv        t2,     %[TMP]  \n\t"
                        "slli      t3,     t3,     1       \n\t"
                        "vle.v     v0,     (t1)    \n\t"
                        "add       t1,     t1,     t3      \n\t"
                        "vle.v     v1,     (t1)    \n\t"
                        "add       t1,     t1,     t3      \n\t"
                        "vle.v     v2,     (t1)    \n\t"
                        "add       t1,     t1,     t3      \n\t"
                        "vle.v     v3,     (t1)    \n\t"
                        "add       t1,     t1,     t3      \n\t"
                        "vsseg4e.v v0,     (t2)    \n\t"
                        :
                        : [LEN] "r"(packn), [SRC] "r"(img0), [TMP] "r"(tmpptr)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "t1", "t2", "t3");

                    img0 += size * packn;
                    tmpptr += packn * 4;
                }
            }
        }

        remain_size_start += nn_size << 2;

        nn_size = (size - remain_size_start) >> 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);

            for (int q = 0; q < inch; q++)
            {
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
                    asm volatile(
                        "mv        t3,     %[LEN]  \n\t"
                        "mv        t1,     %[SRC]  \n\t"
                        "mv        t2,     %[TMP]  \n\t"
                        "slli      t3,     t3,     1       \n\t"
                        "vle.v     v0,     (t1)    \n\t"
                        "add       t1,     t1,     t3      \n\t"
                        "vle.v     v1,     (t1)    \n\t"
                        "add       t1,     t1,     t3      \n\t"
                        "vsseg2e.v v0,     (t2)    \n\t"
                        :
                        : [LEN] "r"(packn), [SRC] "r"(img0), [TMP] "r"(tmpptr)
                        : "cc", "memory", "v0", "v1", "t1", "t2", "t3");

                    img0 += size * packn;
                    tmpptr += packn * 2;
                }
            }
        }

        remain_size_start += nn_size << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);

            for (int q = 0; q < inch; q++)
            {
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
                    vfloat16m1_t _val = vle16_v_f16m1(img0, vl);
                    vse16_v_f16m1(tmpptr, _val, vl);

                    img0 += size * packn;
                    tmpptr += packn;
                }
            }
        }
    }

    int p = 0;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (; p + 2 < outch; p += 3)
    {
        __fp16* outptr0 = top_blob.channel(p + 0);
        __fp16* outptr1 = top_blob.channel(p + 1);
        __fp16* outptr2 = top_blob.channel(p + 2);

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const __fp16* tmpptr = tmp.channel(i / 8);
            const __fp16* kptr0 = kernel.channel(p + 0);
            const __fp16* kptr1 = kernel.channel(p + 1);
            const __fp16* kptr2 = kernel.channel(p + 2);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat16m1_t _sum00 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum01 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum02 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum03 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum04 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum05 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum06 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum07 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum10 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum11 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum12 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum13 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum14 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum15 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum16 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum17 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum20 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum21 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum22 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum23 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum24 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum25 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum26 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum27 = vfmv_v_f_f16m1(0.f, vl);

            if (bias)
            {
                _sum00 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum01 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum02 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum03 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum04 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum05 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum06 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum07 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum10 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum11 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum12 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum13 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum14 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum15 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum16 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum17 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum20 = vle16_v_f16m1(bias + (p + 2) * packn, vl);
                _sum21 = vle16_v_f16m1(bias + (p + 2) * packn, vl);
                _sum22 = vle16_v_f16m1(bias + (p + 2) * packn, vl);
                _sum23 = vle16_v_f16m1(bias + (p + 2) * packn, vl);
                _sum24 = vle16_v_f16m1(bias + (p + 2) * packn, vl);
                _sum25 = vle16_v_f16m1(bias + (p + 2) * packn, vl);
                _sum26 = vle16_v_f16m1(bias + (p + 2) * packn, vl);
                _sum27 = vle16_v_f16m1(bias + (p + 2) * packn, vl);
            }

            for (int j = 0; j < nn; j++)
            {
                __fp16 val0 = *tmpptr++;
                __fp16 val1 = *tmpptr++;
                __fp16 val2 = *tmpptr++;
                __fp16 val3 = *tmpptr++;
                __fp16 val4 = *tmpptr++;
                __fp16 val5 = *tmpptr++;
                __fp16 val6 = *tmpptr++;
                __fp16 val7 = *tmpptr++;
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                vfloat16m1_t _w1 = vle16_v_f16m1(kptr1, vl);
                vfloat16m1_t _w2 = vle16_v_f16m1(kptr2, vl);
                _sum00 = vfmacc_vf_f16m1(_sum00, val0, _w0, vl);
                _sum01 = vfmacc_vf_f16m1(_sum01, val1, _w0, vl);
                _sum02 = vfmacc_vf_f16m1(_sum02, val2, _w0, vl);
                _sum03 = vfmacc_vf_f16m1(_sum03, val3, _w0, vl);
                _sum04 = vfmacc_vf_f16m1(_sum04, val4, _w0, vl);
                _sum05 = vfmacc_vf_f16m1(_sum05, val5, _w0, vl);
                _sum06 = vfmacc_vf_f16m1(_sum06, val6, _w0, vl);
                _sum07 = vfmacc_vf_f16m1(_sum07, val7, _w0, vl);
                _sum10 = vfmacc_vf_f16m1(_sum10, val0, _w1, vl);
                _sum11 = vfmacc_vf_f16m1(_sum11, val1, _w1, vl);
                _sum12 = vfmacc_vf_f16m1(_sum12, val2, _w1, vl);
                _sum13 = vfmacc_vf_f16m1(_sum13, val3, _w1, vl);
                _sum14 = vfmacc_vf_f16m1(_sum14, val4, _w1, vl);
                _sum15 = vfmacc_vf_f16m1(_sum15, val5, _w1, vl);
                _sum16 = vfmacc_vf_f16m1(_sum16, val6, _w1, vl);
                _sum17 = vfmacc_vf_f16m1(_sum17, val7, _w1, vl);
                _sum20 = vfmacc_vf_f16m1(_sum20, val0, _w2, vl);
                _sum21 = vfmacc_vf_f16m1(_sum21, val1, _w2, vl);
                _sum22 = vfmacc_vf_f16m1(_sum22, val2, _w2, vl);
                _sum23 = vfmacc_vf_f16m1(_sum23, val3, _w2, vl);
                _sum24 = vfmacc_vf_f16m1(_sum24, val4, _w2, vl);
                _sum25 = vfmacc_vf_f16m1(_sum25, val5, _w2, vl);
                _sum26 = vfmacc_vf_f16m1(_sum26, val6, _w2, vl);
                _sum27 = vfmacc_vf_f16m1(_sum27, val7, _w2, vl);

                kptr0 += packn;
                kptr1 += packn;
                kptr2 += packn;
            }

            vse16_v_f16m1(outptr0 + packn * 0, _sum00, vl);
            vse16_v_f16m1(outptr0 + packn * 1, _sum01, vl);
            vse16_v_f16m1(outptr0 + packn * 2, _sum02, vl);
            vse16_v_f16m1(outptr0 + packn * 3, _sum03, vl);
            vse16_v_f16m1(outptr0 + packn * 4, _sum04, vl);
            vse16_v_f16m1(outptr0 + packn * 5, _sum05, vl);
            vse16_v_f16m1(outptr0 + packn * 6, _sum06, vl);
            vse16_v_f16m1(outptr0 + packn * 7, _sum07, vl);
            vse16_v_f16m1(outptr1 + packn * 0, _sum10, vl);
            vse16_v_f16m1(outptr1 + packn * 1, _sum11, vl);
            vse16_v_f16m1(outptr1 + packn * 2, _sum12, vl);
            vse16_v_f16m1(outptr1 + packn * 3, _sum13, vl);
            vse16_v_f16m1(outptr1 + packn * 4, _sum14, vl);
            vse16_v_f16m1(outptr1 + packn * 5, _sum15, vl);
            vse16_v_f16m1(outptr1 + packn * 6, _sum16, vl);
            vse16_v_f16m1(outptr1 + packn * 7, _sum17, vl);
            vse16_v_f16m1(outptr2 + packn * 0, _sum20, vl);
            vse16_v_f16m1(outptr2 + packn * 1, _sum21, vl);
            vse16_v_f16m1(outptr2 + packn * 2, _sum22, vl);
            vse16_v_f16m1(outptr2 + packn * 3, _sum23, vl);
            vse16_v_f16m1(outptr2 + packn * 4, _sum24, vl);
            vse16_v_f16m1(outptr2 + packn * 5, _sum25, vl);
            vse16_v_f16m1(outptr2 + packn * 6, _sum26, vl);
            vse16_v_f16m1(outptr2 + packn * 7, _sum27, vl);

            outptr0 += packn * 8;
            outptr1 += packn * 8;
            outptr2 += packn * 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const __fp16* kptr0 = kernel.channel(p + 0);
            const __fp16* kptr1 = kernel.channel(p + 1);
            const __fp16* kptr2 = kernel.channel(p + 2);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat16m1_t _sum00 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum01 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum02 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum03 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum10 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum11 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum12 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum13 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum20 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum21 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum22 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum23 = vfmv_v_f_f16m1(0.f, vl);

            if (bias)
            {
                _sum00 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum01 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum02 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum03 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum10 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum11 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum12 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum13 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum20 = vle16_v_f16m1(bias + (p + 2) * packn, vl);
                _sum21 = vle16_v_f16m1(bias + (p + 2) * packn, vl);
                _sum22 = vle16_v_f16m1(bias + (p + 2) * packn, vl);
                _sum23 = vle16_v_f16m1(bias + (p + 2) * packn, vl);
            }

            for (int j = 0; j < nn; j++)
            {
                __fp16 val0 = *tmpptr++;
                __fp16 val1 = *tmpptr++;
                __fp16 val2 = *tmpptr++;
                __fp16 val3 = *tmpptr++;
                __fp16 val4 = *tmpptr++;
                __fp16 val5 = *tmpptr++;
                __fp16 val6 = *tmpptr++;
                __fp16 val7 = *tmpptr++;
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                vfloat16m1_t _w1 = vle16_v_f16m1(kptr1, vl);
                vfloat16m1_t _w2 = vle16_v_f16m1(kptr2, vl);
                _sum00 = vfmacc_vf_f16m1(_sum00, val0, _w0, vl);
                _sum01 = vfmacc_vf_f16m1(_sum01, val1, _w0, vl);
                _sum02 = vfmacc_vf_f16m1(_sum02, val2, _w0, vl);
                _sum03 = vfmacc_vf_f16m1(_sum03, val3, _w0, vl);
                _sum10 = vfmacc_vf_f16m1(_sum10, val0, _w1, vl);
                _sum11 = vfmacc_vf_f16m1(_sum11, val1, _w1, vl);
                _sum12 = vfmacc_vf_f16m1(_sum12, val2, _w1, vl);
                _sum13 = vfmacc_vf_f16m1(_sum13, val3, _w1, vl);
                _sum20 = vfmacc_vf_f16m1(_sum20, val0, _w2, vl);
                _sum21 = vfmacc_vf_f16m1(_sum21, val1, _w2, vl);
                _sum22 = vfmacc_vf_f16m1(_sum22, val2, _w2, vl);
                _sum23 = vfmacc_vf_f16m1(_sum23, val3, _w2, vl);

                kptr0 += packn;
                kptr1 += packn;
                kptr2 += packn;
            }

            vse16_v_f16m1(outptr0 + packn * 0, _sum00, vl);
            vse16_v_f16m1(outptr0 + packn * 1, _sum01, vl);
            vse16_v_f16m1(outptr0 + packn * 2, _sum02, vl);
            vse16_v_f16m1(outptr0 + packn * 3, _sum03, vl);
            vse16_v_f16m1(outptr1 + packn * 0, _sum10, vl);
            vse16_v_f16m1(outptr1 + packn * 1, _sum11, vl);
            vse16_v_f16m1(outptr1 + packn * 2, _sum12, vl);
            vse16_v_f16m1(outptr1 + packn * 3, _sum13, vl);
            vse16_v_f16m1(outptr2 + packn * 0, _sum20, vl);
            vse16_v_f16m1(outptr2 + packn * 1, _sum21, vl);
            vse16_v_f16m1(outptr2 + packn * 2, _sum22, vl);
            vse16_v_f16m1(outptr2 + packn * 3, _sum23, vl);

            outptr0 += packn * 4;
            outptr1 += packn * 4;
            outptr2 += packn * 4;
        }
        for (; i + 1 < size; i += 2)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const __fp16* kptr0 = kernel.channel(p + 0);
            const __fp16* kptr1 = kernel.channel(p + 1);
            const __fp16* kptr2 = kernel.channel(p + 2);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat16m1_t _sum00 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum01 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum10 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum11 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum20 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum21 = vfmv_v_f_f16m1(0.f, vl);

            if (bias)
            {
                _sum00 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum01 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum10 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum11 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum20 = vle16_v_f16m1(bias + (p + 2) * packn, vl);
                _sum21 = vle16_v_f16m1(bias + (p + 2) * packn, vl);
            }

            for (int j = 0; j < nn; j++)
            {
                __fp16 val0 = *tmpptr++;
                __fp16 val1 = *tmpptr++;
                __fp16 val2 = *tmpptr++;
                __fp16 val3 = *tmpptr++;
                __fp16 val4 = *tmpptr++;
                __fp16 val5 = *tmpptr++;
                __fp16 val6 = *tmpptr++;
                __fp16 val7 = *tmpptr++;
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                vfloat16m1_t _w1 = vle16_v_f16m1(kptr1, vl);
                vfloat16m1_t _w2 = vle16_v_f16m1(kptr2, vl);
                _sum00 = vfmacc_vf_f16m1(_sum00, val0, _w0, vl);
                _sum01 = vfmacc_vf_f16m1(_sum01, val1, _w0, vl);
                _sum10 = vfmacc_vf_f16m1(_sum10, val0, _w1, vl);
                _sum11 = vfmacc_vf_f16m1(_sum11, val1, _w1, vl);
                _sum20 = vfmacc_vf_f16m1(_sum20, val0, _w2, vl);
                _sum21 = vfmacc_vf_f16m1(_sum21, val1, _w2, vl);

                kptr0 += packn;
                kptr1 += packn;
                kptr2 += packn;
            }

            vse16_v_f16m1(outptr0 + packn * 0, _sum00, vl);
            vse16_v_f16m1(outptr0 + packn * 1, _sum01, vl);
            vse16_v_f16m1(outptr1 + packn * 0, _sum10, vl);
            vse16_v_f16m1(outptr1 + packn * 1, _sum11, vl);
            vse16_v_f16m1(outptr2 + packn * 0, _sum20, vl);
            vse16_v_f16m1(outptr2 + packn * 1, _sum21, vl);

            outptr0 += packn * 2;
            outptr1 += packn * 2;
            outptr2 += packn * 2;
        }
        for (; i < size; i++)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const __fp16* kptr0 = kernel.channel(p + 0);
            const __fp16* kptr1 = kernel.channel(p + 1);
            const __fp16* kptr2 = kernel.channel(p + 2);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat16m1_t _sum00 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum10 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum20 = vfmv_v_f_f16m1(0.f, vl);

            if (bias)
            {
                _sum00 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum10 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum20 = vle16_v_f16m1(bias + (p + 2) * packn, vl);
            }

            for (int j = 0; j < nn; j++)
            {
                __fp16 val0 = *tmpptr++;
                __fp16 val1 = *tmpptr++;
                __fp16 val2 = *tmpptr++;
                __fp16 val3 = *tmpptr++;
                __fp16 val4 = *tmpptr++;
                __fp16 val5 = *tmpptr++;
                __fp16 val6 = *tmpptr++;
                __fp16 val7 = *tmpptr++;
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                vfloat16m1_t _w1 = vle16_v_f16m1(kptr1, vl);
                vfloat16m1_t _w2 = vle16_v_f16m1(kptr2, vl);
                _sum00 = vfmacc_vf_f16m1(_sum00, val0, _w0, vl);
                _sum10 = vfmacc_vf_f16m1(_sum10, val0, _w1, vl);
                _sum20 = vfmacc_vf_f16m1(_sum20, val0, _w2, vl);

                kptr0 += packn;
                kptr1 += packn;
                kptr2 += packn;
            }

            vse16_v_f16m1(outptr0 + packn * 0, _sum00, vl);
            vse16_v_f16m1(outptr1 + packn * 0, _sum10, vl);
            vse16_v_f16m1(outptr2 + packn * 0, _sum20, vl);

            outptr0 += packn * 1;
            outptr1 += packn * 1;
            outptr2 += packn * 1;
        }
    }
    #pragma omp parallel for num_threads(opt.num_threads)
    for (; p + 1 < outch; p += 2)
    {
        __fp16* outptr0 = top_blob.channel(p);
        __fp16* outptr1 = top_blob.channel(p + 1);

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const __fp16* tmpptr = tmp.channel(i / 8);
            const __fp16* kptr0 = kernel.channel(p);
            const __fp16* kptr1 = kernel.channel(p + 1);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat16m1_t _sum00 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum01 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum02 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum03 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum04 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum05 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum06 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum07 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum10 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum11 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum12 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum13 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum14 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum15 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum16 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum17 = vfmv_v_f_f16m1(0.f, vl);

            if (bias)
            {
                _sum00 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum01 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum02 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum03 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum04 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum05 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum06 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum07 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum10 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum11 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum12 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum13 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum14 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum15 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum16 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum17 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
            }

            for (int j = 0; j < nn; j++)
            {
                __fp16 val0 = *tmpptr++;
                __fp16 val1 = *tmpptr++;
                __fp16 val2 = *tmpptr++;
                __fp16 val3 = *tmpptr++;
                __fp16 val4 = *tmpptr++;
                __fp16 val5 = *tmpptr++;
                __fp16 val6 = *tmpptr++;
                __fp16 val7 = *tmpptr++;
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                vfloat16m1_t _w1 = vle16_v_f16m1(kptr1, vl);
                _sum00 = vfmacc_vf_f16m1(_sum00, val0, _w0, vl);
                _sum01 = vfmacc_vf_f16m1(_sum01, val1, _w0, vl);
                _sum02 = vfmacc_vf_f16m1(_sum02, val2, _w0, vl);
                _sum03 = vfmacc_vf_f16m1(_sum03, val3, _w0, vl);
                _sum04 = vfmacc_vf_f16m1(_sum04, val4, _w0, vl);
                _sum05 = vfmacc_vf_f16m1(_sum05, val5, _w0, vl);
                _sum06 = vfmacc_vf_f16m1(_sum06, val6, _w0, vl);
                _sum07 = vfmacc_vf_f16m1(_sum07, val7, _w0, vl);
                _sum10 = vfmacc_vf_f16m1(_sum10, val0, _w1, vl);
                _sum11 = vfmacc_vf_f16m1(_sum11, val1, _w1, vl);
                _sum12 = vfmacc_vf_f16m1(_sum12, val2, _w1, vl);
                _sum13 = vfmacc_vf_f16m1(_sum13, val3, _w1, vl);
                _sum14 = vfmacc_vf_f16m1(_sum14, val4, _w1, vl);
                _sum15 = vfmacc_vf_f16m1(_sum15, val5, _w1, vl);
                _sum16 = vfmacc_vf_f16m1(_sum16, val6, _w1, vl);
                _sum17 = vfmacc_vf_f16m1(_sum17, val7, _w1, vl);

                kptr0 += packn;
                kptr1 += packn;
            }

            vse16_v_f16m1(outptr0 + packn * 0, _sum00, vl);
            vse16_v_f16m1(outptr0 + packn * 1, _sum01, vl);
            vse16_v_f16m1(outptr0 + packn * 2, _sum02, vl);
            vse16_v_f16m1(outptr0 + packn * 3, _sum03, vl);
            vse16_v_f16m1(outptr0 + packn * 4, _sum04, vl);
            vse16_v_f16m1(outptr0 + packn * 5, _sum05, vl);
            vse16_v_f16m1(outptr0 + packn * 6, _sum06, vl);
            vse16_v_f16m1(outptr0 + packn * 7, _sum07, vl);
            vse16_v_f16m1(outptr1 + packn * 0, _sum10, vl);
            vse16_v_f16m1(outptr1 + packn * 1, _sum11, vl);
            vse16_v_f16m1(outptr1 + packn * 2, _sum12, vl);
            vse16_v_f16m1(outptr1 + packn * 3, _sum13, vl);
            vse16_v_f16m1(outptr1 + packn * 4, _sum14, vl);
            vse16_v_f16m1(outptr1 + packn * 5, _sum15, vl);
            vse16_v_f16m1(outptr1 + packn * 6, _sum16, vl);
            vse16_v_f16m1(outptr1 + packn * 7, _sum17, vl);

            outptr0 += packn * 8;
            outptr1 += packn * 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const __fp16* kptr0 = kernel.channel(p);
            const __fp16* kptr1 = kernel.channel(p + 1);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat16m1_t _sum00 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum01 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum02 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum03 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum10 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum11 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum12 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum13 = vfmv_v_f_f16m1(0.f, vl);

            if (bias)
            {
                _sum00 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum01 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum02 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum03 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum10 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum11 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum12 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum13 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
            }

            for (int j = 0; j < nn; j++)
            {
                __fp16 val0 = *tmpptr++;
                __fp16 val1 = *tmpptr++;
                __fp16 val2 = *tmpptr++;
                __fp16 val3 = *tmpptr++;
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                vfloat16m1_t _w1 = vle16_v_f16m1(kptr1, vl);
                _sum00 = vfmacc_vf_f16m1(_sum00, val0, _w0, vl);
                _sum01 = vfmacc_vf_f16m1(_sum01, val1, _w0, vl);
                _sum02 = vfmacc_vf_f16m1(_sum02, val2, _w0, vl);
                _sum03 = vfmacc_vf_f16m1(_sum03, val3, _w0, vl);
                _sum10 = vfmacc_vf_f16m1(_sum10, val0, _w1, vl);
                _sum11 = vfmacc_vf_f16m1(_sum11, val1, _w1, vl);
                _sum12 = vfmacc_vf_f16m1(_sum12, val2, _w1, vl);
                _sum13 = vfmacc_vf_f16m1(_sum13, val3, _w1, vl);

                kptr0 += packn;
                kptr1 += packn;
            }

            vse16_v_f16m1(outptr0 + packn * 0, _sum00, vl);
            vse16_v_f16m1(outptr0 + packn * 1, _sum01, vl);
            vse16_v_f16m1(outptr0 + packn * 2, _sum02, vl);
            vse16_v_f16m1(outptr0 + packn * 3, _sum03, vl);
            vse16_v_f16m1(outptr1 + packn * 0, _sum10, vl);
            vse16_v_f16m1(outptr1 + packn * 1, _sum11, vl);
            vse16_v_f16m1(outptr1 + packn * 2, _sum12, vl);
            vse16_v_f16m1(outptr1 + packn * 3, _sum13, vl);

            outptr0 += packn * 4;
            outptr1 += packn * 4;
        }
        for (; i + 1 < size; i += 2)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const __fp16* kptr0 = kernel.channel(p);
            const __fp16* kptr1 = kernel.channel(p + 1);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat16m1_t _sum00 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum01 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum10 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum11 = vfmv_v_f_f16m1(0.f, vl);

            if (bias)
            {
                _sum00 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum01 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum10 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
                _sum11 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
            }

            for (int j = 0; j < nn; j++)
            {
                __fp16 val0 = *tmpptr++;
                __fp16 val1 = *tmpptr++;
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                vfloat16m1_t _w1 = vle16_v_f16m1(kptr1, vl);
                _sum00 = vfmacc_vf_f16m1(_sum00, val0, _w0, vl);
                _sum01 = vfmacc_vf_f16m1(_sum01, val1, _w0, vl);
                _sum10 = vfmacc_vf_f16m1(_sum10, val0, _w1, vl);
                _sum11 = vfmacc_vf_f16m1(_sum11, val1, _w1, vl);

                kptr0 += packn;
                kptr1 += packn;
            }

            vse16_v_f16m1(outptr0 + packn * 0, _sum00, vl);
            vse16_v_f16m1(outptr0 + packn * 1, _sum01, vl);
            vse16_v_f16m1(outptr1 + packn * 0, _sum10, vl);
            vse16_v_f16m1(outptr1 + packn * 1, _sum11, vl);

            outptr0 += packn * 2;
            outptr1 += packn * 2;
        }
        for (; i < size; i++)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const __fp16* kptr0 = kernel.channel(p);
            const __fp16* kptr1 = kernel.channel(p + 1);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat16m1_t _sum00 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum10 = vfmv_v_f_f16m1(0.f, vl);

            if (bias)
            {
                _sum00 = vle16_v_f16m1(bias + (p + 0) * packn, vl);
                _sum10 = vle16_v_f16m1(bias + (p + 1) * packn, vl);
            }

            for (int j = 0; j < nn; j++)
            {
                __fp16 val0 = *tmpptr++;
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                vfloat16m1_t _w1 = vle16_v_f16m1(kptr1, vl);
                _sum00 = vfmacc_vf_f16m1(_sum00, val0, _w0, vl);
                _sum10 = vfmacc_vf_f16m1(_sum10, val0, _w1, vl);

                kptr0 += packn;
                kptr1 += packn;
            }

            vse16_v_f16m1(outptr0 + packn * 0, _sum00, vl);
            vse16_v_f16m1(outptr1 + packn * 0, _sum10, vl);

            outptr0 += packn;
            outptr1 += packn;
        }
    }
    #pragma omp parallel for num_threads(opt.num_threads)
    for (; p < outch; p++)
    {
        __fp16* outptr0 = top_blob.channel(p);

        int i = 0;
        for (; i + 15 < size; i += 16)
        {
            const __fp16* tmpptr0 = tmp.channel(i / 8);
            const __fp16* tmpptr1 = tmp.channel(i / 8 + 1);
            const __fp16* kptr0 = kernel.channel(p);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat16m1_t _sum00 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum01 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum02 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum03 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum04 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum05 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum06 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum07 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum10 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum11 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum12 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum13 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum14 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum15 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum16 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum17 = vfmv_v_f_f16m1(0.f, vl);

            if (bias)
            {
                _sum00 = vle16_v_f16m1(bias + p * packn, vl);
                _sum01 = vle16_v_f16m1(bias + p * packn, vl);
                _sum02 = vle16_v_f16m1(bias + p * packn, vl);
                _sum03 = vle16_v_f16m1(bias + p * packn, vl);
                _sum04 = vle16_v_f16m1(bias + p * packn, vl);
                _sum05 = vle16_v_f16m1(bias + p * packn, vl);
                _sum06 = vle16_v_f16m1(bias + p * packn, vl);
                _sum07 = vle16_v_f16m1(bias + p * packn, vl);
                _sum10 = vle16_v_f16m1(bias + p * packn, vl);
                _sum11 = vle16_v_f16m1(bias + p * packn, vl);
                _sum12 = vle16_v_f16m1(bias + p * packn, vl);
                _sum13 = vle16_v_f16m1(bias + p * packn, vl);
                _sum14 = vle16_v_f16m1(bias + p * packn, vl);
                _sum15 = vle16_v_f16m1(bias + p * packn, vl);
                _sum16 = vle16_v_f16m1(bias + p * packn, vl);
                _sum17 = vle16_v_f16m1(bias + p * packn, vl);
            }

            for (int j = 0; j < nn; j++)
            {
                __fp16 val00 = *tmpptr0++;
                __fp16 val01 = *tmpptr0++;
                __fp16 val02 = *tmpptr0++;
                __fp16 val03 = *tmpptr0++;
                __fp16 val04 = *tmpptr0++;
                __fp16 val05 = *tmpptr0++;
                __fp16 val06 = *tmpptr0++;
                __fp16 val07 = *tmpptr0++;
                __fp16 val10 = *tmpptr1++;
                __fp16 val11 = *tmpptr1++;
                __fp16 val12 = *tmpptr1++;
                __fp16 val13 = *tmpptr1++;
                __fp16 val14 = *tmpptr1++;
                __fp16 val15 = *tmpptr1++;
                __fp16 val16 = *tmpptr1++;
                __fp16 val17 = *tmpptr1++;
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                _sum00 = vfmacc_vf_f16m1(_sum00, val00, _w0, vl);
                _sum01 = vfmacc_vf_f16m1(_sum01, val01, _w0, vl);
                _sum02 = vfmacc_vf_f16m1(_sum02, val02, _w0, vl);
                _sum03 = vfmacc_vf_f16m1(_sum03, val03, _w0, vl);
                _sum04 = vfmacc_vf_f16m1(_sum04, val04, _w0, vl);
                _sum05 = vfmacc_vf_f16m1(_sum05, val05, _w0, vl);
                _sum06 = vfmacc_vf_f16m1(_sum06, val06, _w0, vl);
                _sum07 = vfmacc_vf_f16m1(_sum07, val07, _w0, vl);
                _sum10 = vfmacc_vf_f16m1(_sum10, val10, _w0, vl);
                _sum11 = vfmacc_vf_f16m1(_sum11, val11, _w0, vl);
                _sum12 = vfmacc_vf_f16m1(_sum12, val12, _w0, vl);
                _sum13 = vfmacc_vf_f16m1(_sum13, val13, _w0, vl);
                _sum14 = vfmacc_vf_f16m1(_sum14, val14, _w0, vl);
                _sum15 = vfmacc_vf_f16m1(_sum15, val15, _w0, vl);
                _sum16 = vfmacc_vf_f16m1(_sum16, val16, _w0, vl);
                _sum17 = vfmacc_vf_f16m1(_sum17, val17, _w0, vl);

                kptr0 += packn;
            }

            vse16_v_f16m1(outptr0 + packn * 0x0, _sum00, vl);
            vse16_v_f16m1(outptr0 + packn * 0x1, _sum01, vl);
            vse16_v_f16m1(outptr0 + packn * 0x2, _sum02, vl);
            vse16_v_f16m1(outptr0 + packn * 0x3, _sum03, vl);
            vse16_v_f16m1(outptr0 + packn * 0x4, _sum04, vl);
            vse16_v_f16m1(outptr0 + packn * 0x5, _sum05, vl);
            vse16_v_f16m1(outptr0 + packn * 0x6, _sum06, vl);
            vse16_v_f16m1(outptr0 + packn * 0x7, _sum07, vl);
            vse16_v_f16m1(outptr0 + packn * 0x8, _sum10, vl);
            vse16_v_f16m1(outptr0 + packn * 0x9, _sum11, vl);
            vse16_v_f16m1(outptr0 + packn * 0xa, _sum12, vl);
            vse16_v_f16m1(outptr0 + packn * 0xb, _sum13, vl);
            vse16_v_f16m1(outptr0 + packn * 0xc, _sum14, vl);
            vse16_v_f16m1(outptr0 + packn * 0xd, _sum15, vl);
            vse16_v_f16m1(outptr0 + packn * 0xe, _sum16, vl);
            vse16_v_f16m1(outptr0 + packn * 0xf, _sum17, vl);

            outptr0 += packn * 16;
        }
        for (; i + 7 < size; i += 8)
        {
            const __fp16* tmpptr = tmp.channel(i / 8);
            const __fp16* kptr0 = kernel.channel(p);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat16m1_t _sum0 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum1 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum2 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum3 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum4 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum5 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum6 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum7 = vfmv_v_f_f16m1(0.f, vl);

            if (bias)
            {
                _sum0 = vle16_v_f16m1(bias + p * packn, vl);
                _sum1 = vle16_v_f16m1(bias + p * packn, vl);
                _sum2 = vle16_v_f16m1(bias + p * packn, vl);
                _sum3 = vle16_v_f16m1(bias + p * packn, vl);
                _sum4 = vle16_v_f16m1(bias + p * packn, vl);
                _sum5 = vle16_v_f16m1(bias + p * packn, vl);
                _sum6 = vle16_v_f16m1(bias + p * packn, vl);
                _sum7 = vle16_v_f16m1(bias + p * packn, vl);
            }

            for (int j = 0; j < nn; j++)
            {
                __fp16 val0 = *tmpptr++;
                __fp16 val1 = *tmpptr++;
                __fp16 val2 = *tmpptr++;
                __fp16 val3 = *tmpptr++;
                __fp16 val4 = *tmpptr++;
                __fp16 val5 = *tmpptr++;
                __fp16 val6 = *tmpptr++;
                __fp16 val7 = *tmpptr++;
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, val0, _w0, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, val1, _w0, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, val2, _w0, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, val3, _w0, vl);
                _sum4 = vfmacc_vf_f16m1(_sum4, val4, _w0, vl);
                _sum5 = vfmacc_vf_f16m1(_sum5, val5, _w0, vl);
                _sum6 = vfmacc_vf_f16m1(_sum6, val6, _w0, vl);
                _sum7 = vfmacc_vf_f16m1(_sum7, val7, _w0, vl);

                kptr0 += packn;
            }

            vse16_v_f16m1(outptr0, _sum0, vl);
            vse16_v_f16m1(outptr0 + packn, _sum1, vl);
            vse16_v_f16m1(outptr0 + packn * 2, _sum2, vl);
            vse16_v_f16m1(outptr0 + packn * 3, _sum3, vl);
            vse16_v_f16m1(outptr0 + packn * 4, _sum4, vl);
            vse16_v_f16m1(outptr0 + packn * 5, _sum5, vl);
            vse16_v_f16m1(outptr0 + packn * 6, _sum6, vl);
            vse16_v_f16m1(outptr0 + packn * 7, _sum7, vl);

            outptr0 += packn * 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const __fp16* kptr0 = kernel.channel(p);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat16m1_t _sum0 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum1 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum2 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum3 = vfmv_v_f_f16m1(0.f, vl);

            if (bias)
            {
                _sum0 = vle16_v_f16m1(bias + p * packn, vl);
                _sum1 = vle16_v_f16m1(bias + p * packn, vl);
                _sum2 = vle16_v_f16m1(bias + p * packn, vl);
                _sum3 = vle16_v_f16m1(bias + p * packn, vl);
            }

            for (int j = 0; j < nn; j++)
            {
                __fp16 val0 = *tmpptr++;
                __fp16 val1 = *tmpptr++;
                __fp16 val2 = *tmpptr++;
                __fp16 val3 = *tmpptr++;
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, val0, _w0, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, val1, _w0, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, val2, _w0, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, val3, _w0, vl);

                kptr0 += packn;
            }

            vse16_v_f16m1(outptr0, _sum0, vl);
            vse16_v_f16m1(outptr0 + packn, _sum1, vl);
            vse16_v_f16m1(outptr0 + packn * 2, _sum2, vl);
            vse16_v_f16m1(outptr0 + packn * 3, _sum3, vl);

            outptr0 += packn * 4;
        }
        for (; i + 1 < size; i += 2)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const __fp16* kptr0 = kernel.channel(p);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat16m1_t _sum0 = vfmv_v_f_f16m1(0.f, vl);
            vfloat16m1_t _sum1 = vfmv_v_f_f16m1(0.f, vl);

            if (bias)
            {
                _sum0 = vle16_v_f16m1(bias + p * packn, vl);
                _sum1 = vle16_v_f16m1(bias + p * packn, vl);
            }

            for (int j = 0; j < nn; j++)
            {
                __fp16 val0 = *tmpptr++;
                __fp16 val1 = *tmpptr++;
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, val0, _w0, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, val1, _w0, vl);

                kptr0 += packn;
            }

            vse16_v_f16m1(outptr0, _sum0, vl);
            vse16_v_f16m1(outptr0 + packn, _sum1, vl);

            outptr0 += packn * 2;
        }
        for (; i < size; i++)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const __fp16* kptr0 = kernel.channel(p);

            int nn = inch * maxk * packn; // inch always > 0

            vfloat16m1_t _sum = vfmv_v_f_f16m1(0.f, vl);

            if (bias)
            {
                _sum = vle16_v_f16m1(bias + p * packn, vl);
            }

            for (int j = 0; j < nn; j++)
            {
                __fp16 val = *tmpptr++;
                vfloat16m1_t _w0 = vle16_v_f16m1(kptr0, vl);
                _sum = vfmacc_vf_f16m1(_sum, val, _w0, vl);

                kptr0 += packn;
            }

            vse16_v_f16m1(outptr0, _sum, vl);

            outptr0 += packn;
        }
    }
}

static void convolution_im2col_sgemm_packn_fp16sa_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 2u * packn, packn, opt.workspace_allocator);
    {
        const int gap = (w * stride_h - outw * stride_w) * packn;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            __fp16* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const __fp16* sptr = img.row<const __fp16>(dilation_h * u) + dilation_w * v * packn;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j < outw; j++)
                        {
                            vfloat16m1_t _val = vle16_v_f16m1(sptr, vl);
                            vse16_v_f16m1(ptr, _val, vl);

                            sptr += stride_w * packn;
                            ptr += packn;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_packn_fp16sa_rvv(bottom_im2col, top_blob, kernel, _bias, opt);
}
