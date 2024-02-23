// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

// #if !(__ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD)
// #if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
// void convolution_transform_kernel_packed_int8_i8mm(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);
// void convolution_packed_int8_i8mm(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt);
// #endif

// #if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
// void convolution_transform_kernel_packed_int8_asimddp(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);
// void convolution_packed_int8_asimddp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt);
// #endif
// #endif

static void convolution_transform_kernel_packed_int8(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb

    // clang-format off
    // *INDENT-OFF*
#if __riscv_vector
    if (outch >= 8)
    {
        if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + inch % 8, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)64u, 64);
        else
            kernel_tm.create(maxk, inch, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)8u, 8);
    }
    else if (outch >= 4)
    {
        if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + inch % 8, outch / 4 + (outch % 4) / 2 + outch % 2, (size_t)32u, 32);
        else
            kernel_tm.create(maxk, inch, outch / 4 + (outch % 4) / 2 + outch % 2, (size_t)4u, 4);
    }
    else
#endif // __riscv_vector
    if (outch >= 2)
    {
#if __riscv_vector
        if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + inch % 8, outch / 2 + outch % 2, (size_t)16u, 16);
        else
#endif // __riscv_vector
            kernel_tm.create(maxk, inch, outch / 2 + outch % 2, (size_t)2u, 2);
    }
    else
    {
#if __riscv_vector
        if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + inch % 8, outch, (size_t)8u, 8);
        else
#endif // __riscv_vector
            kernel_tm.create(maxk, inch, outch, (size_t)1u, 1);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __riscv_vector
    for (; q + 7 < outch; q += 8)
    {
        const signed char* kptr0 = (const signed char*)kernel + q * inch * maxk;
        const signed char* kptr1 = (const signed char*)kernel + (q + 1) * inch * maxk;
        const signed char* kptr2 = (const signed char*)kernel + (q + 2) * inch * maxk;
        const signed char* kptr3 = (const signed char*)kernel + (q + 3) * inch * maxk;
        const signed char* kptr4 = (const signed char*)kernel + (q + 4) * inch * maxk;
        const signed char* kptr5 = (const signed char*)kernel + (q + 5) * inch * maxk;
        const signed char* kptr6 = (const signed char*)kernel + (q + 6) * inch * maxk;
        const signed char* kptr7 = (const signed char*)kernel + (q + 7) * inch * maxk;

        signed char* g00 = kernel_tm.channel(q / 8);

        int p = 0;
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;
                const signed char* k2 = kptr2 + k;
                const signed char* k3 = kptr3 + k;
                const signed char* k4 = kptr4 + k;
                const signed char* k5 = kptr5 + k;
                const signed char* k6 = kptr6 + k;
                const signed char* k7 = kptr7 + k;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k0[maxk];
                    g00[2] = k1[0];
                    g00[3] = k1[maxk];
                    g00[4] = k2[0];
                    g00[5] = k2[maxk];
                    g00[6] = k3[0];
                    g00[7] = k3[maxk];
                    g00[8] = k4[0];
                    g00[9] = k4[maxk];
                    g00[10] = k5[0];
                    g00[11] = k5[maxk];
                    g00[12] = k6[0];
                    g00[13] = k6[maxk];
                    g00[14] = k7[0];
                    g00[15] = k7[maxk];
                    g00 += 16;
                    k0 += maxk * 2;
                    k1 += maxk * 2;
                    k2 += maxk * 2;
                    k3 += maxk * 2;
                    k4 += maxk * 2;
                    k5 += maxk * 2;
                    k6 += maxk * 2;
                    k7 += maxk * 2;
                }
            }

            kptr0 += maxk * 8;
            kptr1 += maxk * 8;
            kptr2 += maxk * 8;
            kptr3 += maxk * 8;
            kptr4 += maxk * 8;
            kptr5 += maxk * 8;
            kptr6 += maxk * 8;
            kptr7 += maxk * 8;
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;
                const signed char* k2 = kptr2 + k;
                const signed char* k3 = kptr3 + k;
                const signed char* k4 = kptr4 + k;
                const signed char* k5 = kptr5 + k;
                const signed char* k6 = kptr6 + k;
                const signed char* k7 = kptr7 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00[2] = k2[0];
                g00[3] = k3[0];
                g00[4] = k4[0];
                g00[5] = k5[0];
                g00[6] = k6[0];
                g00[7] = k7[0];
                g00 += 8;
            }

            kptr0 += maxk;
            kptr1 += maxk;
            kptr2 += maxk;
            kptr3 += maxk;
            kptr4 += maxk;
            kptr5 += maxk;
            kptr6 += maxk;
            kptr7 += maxk;
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        const signed char* kptr0 = (const signed char*)kernel + q * inch * maxk;
        const signed char* kptr1 = (const signed char*)kernel + (q + 1) * inch * maxk;
        const signed char* kptr2 = (const signed char*)kernel + (q + 2) * inch * maxk;
        const signed char* kptr3 = (const signed char*)kernel + (q + 3) * inch * maxk;

        signed char* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);

        int p = 0;
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;
                const signed char* k2 = kptr2 + k;
                const signed char* k3 = kptr3 + k;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k0[maxk];
                    g00[2] = k1[0];
                    g00[3] = k1[maxk];
                    g00[4] = k2[0];
                    g00[5] = k2[maxk];
                    g00[6] = k3[0];
                    g00[7] = k3[maxk];
                    g00 += 8;
                    k0 += maxk * 2;
                    k1 += maxk * 2;
                    k2 += maxk * 2;
                    k3 += maxk * 2;
                }
            }

            kptr0 += maxk * 8;
            kptr1 += maxk * 8;
            kptr2 += maxk * 8;
            kptr3 += maxk * 8;
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;
                const signed char* k2 = kptr2 + k;
                const signed char* k3 = kptr3 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00[2] = k2[0];
                g00[3] = k3[0];
                g00 += 4;
            }

            kptr0 += maxk;
            kptr1 += maxk;
            kptr2 += maxk;
            kptr3 += maxk;
        }
    }
#endif // __riscv_vector
    for (; q + 1 < outch; q += 2)
    {
        const signed char* kptr0 = (const signed char*)kernel + q * inch * maxk;
        const signed char* kptr1 = (const signed char*)kernel + (q + 1) * inch * maxk;

#if __riscv_vector
        signed char* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2);
#else
        signed char* g00 = kernel_tm.channel(q / 2);
#endif

        int p = 0;
#if __riscv_vector
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[0];
                    k0 += maxk;
                    g00 += 1;
                }
                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k1[0];
                    k1 += maxk;
                    g00 += 1;
                }

                for (int i = 4; i < 8; i++)
                {
                    g00[0] = k0[0];
                    k0 += maxk;
                    g00 += 1;
                }
                for (int i = 4; i < 8; i++)
                {
                    g00[0] = k1[0];
                    k1 += maxk;
                    g00 += 1;
                }
            }

            kptr0 += maxk * 8;
            kptr1 += maxk * 8;
        }
#endif // __riscv_vector
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00 += 2;
            }

            kptr0 += maxk;
            kptr1 += maxk;
        }
    }
    for (; q < outch; q++)
    {
        const signed char* kptr = (const signed char*)kernel + q * inch * maxk;

#if __riscv_vector
        signed char* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
#else
        signed char* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __riscv_vector
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr + k;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[0];
                    k0 += maxk;
                    g00 += 1;
                }
            }

            kptr += maxk * 8;
        }
#endif // __riscv_vector
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr + k;
                g00[0] = k0[0];
                g00++;
            }

            kptr += maxk;
        }
    }
}

static void convolution_packed_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    // #if !(__ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD)
    // #if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    //     if (ncnn::cpu_support_arm_i8mm())
    //     {
    //         convolution_packed_int8_i8mm(bottom_blob, top_blob, weight_data_tm, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
    //         return;
    //     }
    // #endif

    // #if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
    //     if (ncnn::cpu_support_arm_asimddp())
    //     {
    //         convolution_packed_int8_asimddp(bottom_blob, top_blob, weight_data_tm, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
    //         return;
    //     }
    // #endif
    // #endif
    int vl;

    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int inch = bottom_blob.c * elempack;

    const int N = bottom_blob.cstep * elempack;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int outch = top_blob.c * out_elempack;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2 * elempack;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    int nn_outch = 0;
    int remain_outch_start = 0;
#if __riscv_vector
    nn_outch = (outch - remain_outch_start) / 8;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 8;

        // shadowed variable for less openmp task args
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int N = bottom_blob.cstep * elempack;
        const int M = top_blob.cstep * out_elempack;

        int* outptr = top_blob.channel(p / out_elempack);

        int ij = 0;

        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            // int32x4_t _sum0 = vdupq_n_s32(0);
            // int32x4_t _sum1 = vdupq_n_s32(0);
            // int32x4_t _sum2 = vdupq_n_s32(0);
            // int32x4_t _sum3 = vdupq_n_s32(0);

            vl = 8;
            vint32m2_t _sum01 = vmv_v_x_i32m2(0, vl);
            // vint32m2_t _sum23 = vmv_v_x_i32m2(0, vl);

            const signed char* kptr = weight_data_tm.channel(p / 8);

            int q = 0;
            {
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        vl = 8;
                        const signed char* r0s = r0 + space_ofs[k];

                        // int8x8_t _r0;
                        vint8m1_t _r0;
                        if (elempack == 8)
                        {
                            _r0 = vle8_v_i8m1(r0s, vl);
                            // _r0 = vld1_s8(r0s);
                        }
                        else // if (elempack == 1)
                        {
                            _r0 = vlse8_v_i8m1(r0s, N * sizeof(signed char), vl);
                            // signed char tmp[8] = {r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7]};
                            // _r0 = vld1_s8(tmp);
                        }

                        // int8x16_t _w0 = vld1q_s8(kptr);
                        // int8x16_t _w1 = vld1q_s8(kptr + 16);
                        // int8x16_t _w2 = vld1q_s8(kptr + 32);
                        // int8x16_t _w3 = vld1q_s8(kptr + 48);
                        vl = 16;
                        vint8m1_t _w0 = vle8_v_i8m1(kptr, vl);
                        vint8m1_t _w1 = vle8_v_i8m1(kptr + 16, vl);
                        vint8m1_t _w2 = vle8_v_i8m1(kptr + 32, vl);
                        vint8m1_t _w3 = vle8_v_i8m1(kptr + 48, vl);

                        vl = 8;

                        // int16x4_t _rr0 = vreinterpret_s16_s8(_r0);
                        vint16m1_t _rr0 = vreinterpret_v_i8m1_i16m1(_r0);

                        vint8m1_t _r0ll = vreinterpret_v_i16m1_i8m1(vrgather_vx_i16m1(_rr0, 0, vl));
                        vint8m1_t _r0lh = vreinterpret_v_i16m1_i8m1(vrgather_vx_i16m1(_rr0, 1, vl));
                        vint8m1_t _r0hl = vreinterpret_v_i16m1_i8m1(vrgather_vx_i16m1(_rr0, 2, vl));
                        vint8m1_t _r0hh = vreinterpret_v_i16m1_i8m1(vrgather_vx_i16m1(_rr0, 3, vl));

                        // uint8_t mask[8] = {8, 9, 10, 11, 12, 13, 14, 15};
                        // vuint8m1_t _index = vle8_v_u8m1(mask, vl);

                        // int8x8_t _r0ll = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 0));
                        // int8x8_t _r0lh = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 1));
                        // int8x8_t _r0hl = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 2));
                        // int8x8_t _r0hh = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 3));

                        vint16m2_t _s0l_m2 = vwmul_vv_i16m2(_r0ll, _w0, vl);
                        vint16m2_t _s1l_m2 = vwmul_vv_i16m2(_r0ll, vslidedown_vx_i8m1(_w0, _w0, 8, vl), vl);
                        vint16m2_t _s0h_m2 = vwmul_vv_i16m2(_r0lh, _w1, vl);
                        vint16m2_t _s1h_m2 = vwmul_vv_i16m2(_r0lh, vslidedown_vx_i8m1(_w1, _w1, 8, vl), vl);

                        // int16x8_t _s0l = vmull_s8(_r0ll, vget_low_s8(_w0));
                        // int16x8_t _s1l = vmull_s8(_r0ll, vget_high_s8(_w0));
                        // int16x8_t _s0h = vmull_s8(_r0lh, vget_low_s8(_w1));
                        // int16x8_t _s1h = vmull_s8(_r0lh, vget_high_s8(_w1));

                        // vint16m1_t _s0l = vget_v_i16m2_i16m2(vwmacc_vv_i16m2(_s0l_m2, _r0hl, _w2, vl), 0);
                        // vint16m1_t _s1l = vget_v_i16m2_i16m2(vwmacc_vv_i16m2(_s1l_m2, _r0hl, vrgather_vv_i8m1(_w2, _index, vl), vl), 0);
                        // vint16m1_t _s2l = vget_v_i16m2_i16m2(vwmacc_vv_i16m2(_s0h_m2, _r0hh, _w3, vl), 0);
                        // vint16m1_t _s3l = vget_v_i16m2_i16m2(vwmacc_vv_i16m2(_s1h_m2, _r0hh, vrgather_vv_i8m1(_w3, _index, vl), vl), 0);

                        _s0l_m2 = vwmacc_vv_i16m2(_s0l_m2, _r0hl, _w2, vl);
                        _s1l_m2 = vwmacc_vv_i16m2(_s1l_m2, _r0hl, vslidedown_vx_i8m1(_w2, _w2, 8, vl), vl);
                        _s0h_m2 = vwmacc_vv_i16m2(_s0h_m2, _r0hh, _w3, vl);
                        _s1h_m2 = vwmacc_vv_i16m2(_s1h_m2, _r0hh, vslidedown_vx_i8m1(_w3, _w3, 8, vl), vl);

                        // _s0l = vmlal_s8(_s0l, _r0hl, vget_low_s8(_w2));
                        // _s1l = vmlal_s8(_s1l, _r0hl, vget_high_s8(_w2));
                        // _s0h = vmlal_s8(_s0h, _r0hh, vget_low_s8(_w3));
                        // _s1h = vmlal_s8(_s1h, _r0hh, vget_high_s8(_w3));

                        vint16m2_t _s01l = vset_v_i16m1_i16m2(_s0l_m2, 1, vget_v_i16m2_i16m1(_s1l_m2, 0));
                        vint16m2_t _s01h = vset_v_i16m1_i16m2(_s0h_m2, 1, vget_v_i16m2_i16m1(_s1h_m2, 0));
                        uint16_t odd_index[8] = {1, 3, 5, 7, 9, 11, 13, 15};
                        uint16_t even_index[8] = {0, 2, 4, 6, 8, 10, 12, 14};
                        vuint16m2_t _odd_index = vle16_v_u16m2(odd_index, vl);
                        vuint16m2_t _even_index = vle16_v_u16m2(even_index, vl);

                        _sum01 = vwadd_wv_i32m2(_sum01, vget_v_i16m2_i16m1(vrgather_vv_i16m2(_s01l, _odd_index, vl), 0), vl);
                        _sum01 = vwadd_wv_i32m2(_sum01, vget_v_i16m2_i16m1(vrgather_vv_i16m2(_s01l, _even_index, vl), 0), vl);
                        _sum01 = vwadd_wv_i32m2(_sum01, vget_v_i16m2_i16m1(vrgather_vv_i16m2(_s01h, _odd_index, vl), 0), vl);
                        _sum01 = vwadd_wv_i32m2(_sum01, vget_v_i16m2_i16m1(vrgather_vv_i16m2(_s01h, _even_index, vl), 0), vl);

                        // _sum0 = vpadalq_s16(_sum0, _s0l);
                        // _sum1 = vpadalq_s16(_sum1, _s1l);
                        // _sum2 = vpadalq_s16(_sum2, _s0h);
                        // _sum3 = vpadalq_s16(_sum3, _s1h);

                        kptr += 64;
                    }
                }

                {
                    // _sum0 = vaddq_s32(_sum0, _sum2);
                    // _sum1 = vaddq_s32(_sum1, _sum3);
                    // _sum01 = vadd_vv_i32m2(_sum01, _sum23, vl);
                }
            }
            for (; q < inch; q++)
            {
                vl = 8;
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    vl = 8;
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        vint8m1_t _val = vmv_v_x_i8m1(r0s[0], vl);
                        vint8m1_t _w = vle8_v_i8m1(kptr, vl);
                        vint16m1_t _s0 = vget_v_i16m2_i16m1(vwmul_vv_i16m2(_val, _w, vl), 0);
                        _sum01 = vwadd_wv_i32m2(_sum01, _s0, vl);
                        // int8x8_t _val = vdup_n_s8(r0s[0]);
                        // int8x8_t _w = vld1_s8(kptr);
                        // int16x8_t _s0 = vmull_s8(_val, _w);
                        // _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                        // _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                        kptr += 8;
                    }
                }
            }
            vl = 8;

            if (out_elempack == 8)
            {
                // vst1q_s32(outptr, _sum0);
                // vst1q_s32(outptr + 4, _sum1);
                vse32_v_i32m2(outptr, _sum01, vl);
                outptr += 8;
            }
            if (out_elempack == 4)
            {
                // vst1q_s32(outptr, _sum0);
                // vst1q_s32(outptr + M, _sum1);
                vl = 4;
                vse32_v_i32m1(outptr, vget_v_i32m2_i32m1(_sum01, 0), vl);
                vse32_v_i32m1(outptr + M, vget_v_i32m2_i32m1(_sum01, 1), vl);
                outptr += 4;
            }
            if (out_elempack == 1)
            {
                vsse32_v_i32m2(outptr, M * sizeof(int), _sum01, vl);
                // outptr[0] = vgetq_lane_s32(_sum0, 0);
                // outptr[M] = vgetq_lane_s32(_sum0, 1);
                // outptr[M * 2] = vgetq_lane_s32(_sum0, 2);
                // outptr[M * 3] = vgetq_lane_s32(_sum0, 3);
                // outptr[M * 4] = vgetq_lane_s32(_sum1, 0);
                // outptr[M * 5] = vgetq_lane_s32(_sum1, 1);
                // outptr[M * 6] = vgetq_lane_s32(_sum1, 2);
                // outptr[M * 7] = vgetq_lane_s32(_sum1, 3);
                outptr += 1;
            }
        }
    }
    remain_outch_start += nn_outch * 8;
    nn_outch = (outch - remain_outch_start) / 4;
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 4;

        // shadowed variable for less openmp task args
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int N = bottom_blob.cstep * elempack;
        const int M = top_blob.cstep * out_elempack;

        int* outptr = top_blob.channel(p / out_elempack);

        int ij = 0;

        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;
            vl = 4;

            vint32m2_t _sum01 = vmv_v_x_i32m2(0, vl);
            // int32x4_t _sum0 = vdupq_n_s32(0);
            // int32x4_t _sum1 = vdupq_n_s32(0);

            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4);

            int q = 0;
            {
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        vl = 8;
                        const signed char* r0s = r0 + space_ofs[k];

                        // int8x8_t _r0;
                        vint8m1_t _r0;
                        if (elempack == 8)
                        {
                            _r0 = vle8_v_i8m1(r0s, vl);
                            // _r0 = vld1_s8(r0s);
                        }
                        else // if (elempack == 1)
                        {
                            // signed char tmp[8] = {r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7]};
                            // _r0 = vld1_s8(tmp);
                            _r0 = vlse8_v_i8m1(r0s, N * sizeof(signed char), vl);
                        }

                        // int8x16_t _w0 = vld1q_s8(kptr);
                        // int8x16_t _w1 = vld1q_s8(kptr + 16);
                        vl = 16;
                        vint8m1_t _w0 = vle8_v_i8m1(kptr, vl);
                        vint8m1_t _w1 = vle8_v_i8m1(kptr + 16, vl);
                        vl = 8;

                        // int16x4_t _rr0 = vreinterpret_s16_s8(_r0);
                        vint16m1_t _rr0 = vreinterpret_v_i8m1_i16m1(_r0);

                        vint8m1_t _r0ll = vreinterpret_v_i16m1_i8m1(vrgather_vx_i16m1(_rr0, 0, vl));
                        vint8m1_t _r0lh = vreinterpret_v_i16m1_i8m1(vrgather_vx_i16m1(_rr0, 1, vl));
                        vint8m1_t _r0hl = vreinterpret_v_i16m1_i8m1(vrgather_vx_i16m1(_rr0, 2, vl));
                        vint8m1_t _r0hh = vreinterpret_v_i16m1_i8m1(vrgather_vx_i16m1(_rr0, 3, vl));

                        vint16m2_t _sl_m2 = vwmul_vv_i16m2(_r0ll, _w0, vl);
                        vint16m2_t _sh_m2 = vwmul_vv_i16m2(_r0lh, vslidedown_vx_i8m1(_w0, _w0, 8, vl), vl);
                        _sl_m2 = vwmacc_vv_i16m2(_sl_m2, _r0hl, _w1, vl);
                        _sh_m2 = vwmacc_vv_i16m2(_sh_m2, _r0hh, vslidedown_vx_i8m1(_w1, _w1, 8, vl), vl);

                        vint16m1_t _sl = vget_v_i16m2_i16m1(_sl_m2, 0);
                        vint16m1_t _sh = vget_v_i16m2_i16m1(_sh_m2, 0);

                        // int8x8_t _r0ll = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 0));
                        // int8x8_t _r0lh = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 1));
                        // int8x8_t _r0hl = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 2));
                        // int8x8_t _r0hh = vreinterpret_s8_s16(vdup_lane_s16(_rr0, 3));

                        // int16x8_t _sl = vmull_s8(_r0ll, vget_low_s8(_w0));
                        // int16x8_t _sh = vmull_s8(_r0lh, vget_high_s8(_w0));
                        // _sl = vmlal_s8(_sl, _r0hl, vget_low_s8(_w1));
                        // _sh = vmlal_s8(_sh, _r0hh, vget_high_s8(_w1));
                        vl = 4;

                        uint16_t odd_index[4] = {1, 3, 5, 7};
                        uint16_t even_index[4] = {0, 2, 4, 6};
                        vuint16m1_t _odd_index = vle16_v_u16m1(odd_index, vl);
                        vuint16m1_t _even_index = vle16_v_u16m1(even_index, vl);

                        _sum01 = vwadd_wv_i32m2(_sum01, vrgather_vv_i16m1(_sl, _odd_index, vl), vl);
                        _sum01 = vwadd_wv_i32m2(_sum01, vrgather_vv_i16m1(_sl, _even_index, vl), vl);
                        _sum01 = vwadd_wv_i32m2(_sum01, vrgather_vv_i16m1(_sh, _odd_index, vl), vl);
                        _sum01 = vwadd_wv_i32m2(_sum01, vrgather_vv_i16m1(_sh, _even_index, vl), vl);

                        // _sum0 = vpadalq_s16(_sum0, _sl);
                        // _sum1 = vpadalq_s16(_sum1, _sh);

                        kptr += 32;
                    }
                }
                // {
                //     _sum0 = vaddq_s32(_sum0, _sum1);
                // }
            }
            for (; q < inch; q++)
            {
                vl = 4;
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    vl = 4;
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        vint8m1_t _val = vmv_v_x_i8m1(r0s[0], vl);
                        vint8m1_t _w = vle8_v_i8m1(kptr, vl);
                        vint16m1_t _s0 = vget_v_i16m2_i16m1(vwmul_vv_i16m2(_val, _w, vl), 0);
                        _sum01 = vwadd_wv_i32m2(_sum01, _s0, vl);
                        // int8x8_t _val = vdup_n_s8(r0s[0]);
                        // int8x8_t _w = vld1_s8(kptr);
                        // int16x8_t _s0 = vmull_s8(_val, _w);
                        // _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));

                        kptr += 4;
                    }
                }
            }
            vl = 4;

            if (out_elempack == 4)
            {
                // vst1q_s32(outptr, _sum0);
                vse32_v_i32m2(outptr, _sum01, vl);
                outptr += 4;
            }
            if (out_elempack == 1)
            {
                vsse32_v_i32m2(outptr, M * sizeof(int), _sum01, vl);
                // outptr[0] = vgetq_lane_s32(_sum0, 0);
                // outptr[M] = vgetq_lane_s32(_sum0, 1);
                // outptr[M * 2] = vgetq_lane_s32(_sum0, 2);
                // outptr[M * 3] = vgetq_lane_s32(_sum0, 3);
                outptr += 1;
            }
        }
    }
    remain_outch_start += nn_outch * 4;
    nn_outch = (outch - remain_outch_start) / 2;
#else // __riscv_vector
    nn_outch = (outch - remain_outch_start) / 2;
    #pragma omp parallel for num_threads(opt.num_threads)
#endif // __riscv_vector
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 2;

        // shadowed variable for less openmp task args
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int N = bottom_blob.cstep * elempack;

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p + 1);

        int ij = 0;

        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            int sum0 = 0;
            int sum1 = 0;

#if __riscv_vector
            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2);
#else
            const signed char* kptr = weight_data_tm.channel(p / 2);
#endif

            int q = 0;
#if __riscv_vector
            {
                // int32x4_t _sum01 = vdupq_n_s32(0);
                vl = 4;
                vint32m2_t _sum01 = vmv_v_x_i32m2(0, vl);
                for (; q + 7 < inch; q += 8)
                {
                    vl = 8;
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        vl = 8;
                        const signed char* r0s = r0 + space_ofs[k];

                        // int8x8_t _r0;
                        vint8m1_t _r0;
                        if (elempack == 8)
                        {
                            _r0 = vle8_v_i8m1(r0s, vl);
                            // _r0 = vld1_s8(r0s);
                        }
                        else // if (elempack == 1)
                        {
                            _r0 = vlse8_v_i8m1(r0s, N * sizeof(signed char), vl);
                            // signed char tmp[8] = {r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7]};
                            // _r0 = vld1_s8(tmp);
                        }

                        // int8x16_t _w0 = vld1q_s8(kptr);
                        vl = 16;
                        vint8m1_t _w0 = vle8_v_i8m1(kptr, vl);
                        vl = 8;
                        // fprintf(stderr, "r0: \n");
                        // print_vint8m1(_r0, 8);
                        vint8m1_t _r0l = vslideup_vx_i8m1(_r0, _r0, 4, vl);
                        vint8m1_t _r0h = vslidedown_vx_i8m1(_r0, _r0, 4, vl);
                        _r0h = vslideup_vx_i8m1(_r0h, _r0h, 4, vl);

                        // vint32m1_t _r0_i16 = vreinterpret_v_i32m1_i8m1(_r0);

                        // int32x2x2_t _rr0 = vzip_s32(vreinterpret_s32_s8(_r0), vreinterpret_s32_s8(_r0));
                        // int8x8_t _r0l = vreinterpret_s8_s32(_rr0.val[0]);
                        // int8x8_t _r0h = vreinterpret_s8_s32(_rr0.val[1]);

                        vint16m2_t _s01_m2 = vwmul_vv_i16m2(_r0l, _w0, vl);
                        _s01_m2 = vwmacc_vv_i16m2(_s01_m2, _r0h, vslidedown_vx_i8m1(_w0, _w0, 8, vl), vl);
                        vint16m1_t _s01 = vget_v_i16m2_i16m1(_s01_m2, 0);

                        vl = 4;
                        uint16_t odd_index[4] = {1, 3, 5, 7};
                        uint16_t even_index[4] = {0, 2, 4, 6};
                        vuint16m1_t _odd_index = vle16_v_u16m1(odd_index, vl);
                        vuint16m1_t _even_index = vle16_v_u16m1(even_index, vl);
                        _sum01 = vwadd_wv_i32m2(_sum01, vrgather_vv_i16m1(_s01, _odd_index, vl), vl);
                        _sum01 = vwadd_wv_i32m2(_sum01, vrgather_vv_i16m1(_s01, _even_index, vl), vl);

                        // int16x8_t _s01 = vmull_s8(_r0l, vget_low_s8(_w0));
                        // _s01 = vmlal_s8(_s01, _r0h, vget_high_s8(_w0));
                        // _sum01 = vpadalq_s16(_sum01, _s01);

                        kptr += 16;
                    }
                }
                int res[4] = {0, 0, 0, 0};
                vl = 4;
                vse32_v_i32m2(res, _sum01, vl);
                sum0 += (res[0] + res[1]);
                sum1 += (res[2] + res[3]);
                // int32x2_t _s0 = vpadd_s32(vget_low_s32(_sum01), vget_high_s32(_sum01));
                // sum0 += vget_lane_s32(_s0, 0);
                // sum1 += vget_lane_s32(_s0, 1);
            }
#endif // __riscv_vector
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum0 += r0s[0] * kptr[0];
                        sum1 += r0s[0] * kptr[1];

                        kptr += 2;
                    }
                }
            }

            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr0 += 1;
            outptr1 += 1;
        }
    }
    remain_outch_start += nn_outch * 2;
    for (int p = remain_outch_start; p < outch; p++)
    {
        int* outptr = top_blob.channel(p);

        int ij = 0;

        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            int sum = 0;

#if __riscv_vector
            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#else
            const signed char* kptr = weight_data_tm.channel(p / 2 + p % 2);
#endif

            int q = 0;
#if __riscv_vector
            {
                vl = 8;
                vint32m2_t _sum01 = vmv_v_x_i32m2(0, vl);
                // int32x4_t _sum0 = vdupq_n_s32(0);
                // int32x4_t _sum1 = vdupq_n_s32(0);
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        vl = 8;
                        const signed char* r0s = r0 + space_ofs[k];

                        vint8m1_t _r0;
                        // int8x8_t _r0;
                        if (elempack == 8)
                        {
                            // _r0 = vld1_s8(r0s);
                            _r0 = vle8_v_i8m1(r0s, vl);
                        }
                        else // if (elempack == 1)
                        {
                            _r0 = vlse8_v_i8m1(r0s, N * sizeof(signed char), vl);
                            // signed char tmp[8] = {r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7]};
                            // _r0 = vld1_s8(tmp);
                        }

                        vint8m1_t _w = vle8_v_i8m1(kptr, vl);
                        vint16m1_t _s0 = vget_v_i16m2_i16m1(vwmul_vv_i16m2(_r0, _w, vl), 0);
                        _sum01 = vwadd_wv_i32m2(_sum01, _s0, vl);
                        // int8x8_t _w = vld1_s8(kptr);

                        // int16x8_t _s0 = vmull_s8(_r0, _w);

                        // _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                        // _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                        kptr += 8;
                    }
                }
                // int32x4_t _sum = vaddq_s32(_sum0, _sum1);
                // #if __aarch64__
                vl = 8;
                vint32m1_t _scalar_sum = vmv_s_x_i32m1(vint32m1_t(), sum, vl);
                sum = vmv_x_s_i32m1_i32(vredsum_vs_i32m2_i32m1(_scalar_sum, _sum01, _scalar_sum, vl));
                // int res[8] = {0, 0, 0, 0};
                // vl = 4;
                // vse32_v_i32m2(res, _sum01, vl);
                // sum += (res[0] + res[1] + res[2] + res[3]);
                // sum += vaddvq_s32(_sum);
                // #else
                //                 int32x2_t _ss = vadd_s32(vget_low_s32(_sum), vget_high_s32(_sum));
                //                 _ss = vpadd_s32(_ss, _ss);
                //                 sum += vget_lane_s32(_ss, 0);
                // #endif
            }
#endif // __riscv_vector
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum += r0s[0] * kptr[0];

                        kptr += 1;
                    }
                }
            }

            outptr[0] = sum;
            outptr += 1;
        }
    }
}
