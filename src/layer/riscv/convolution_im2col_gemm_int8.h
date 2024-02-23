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

static void convolution_im2col_pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    // A = (pa, maxk, inch/pa), outch
    const int A_hstep = A.w;

    signed char* pp = AT;

    int ii = 0;
#if __riscv_vector
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;
        const signed char* p2 = (const signed char*)A + (i + ii + 2) * A_hstep + k;
        const signed char* p3 = (const signed char*)A + (i + ii + 3) * A_hstep + k;
        const signed char* p4 = (const signed char*)A + (i + ii + 4) * A_hstep + k;
        const signed char* p5 = (const signed char*)A + (i + ii + 5) * A_hstep + k;
        const signed char* p6 = (const signed char*)A + (i + ii + 6) * A_hstep + k;
        const signed char* p7 = (const signed char*)A + (i + ii + 7) * A_hstep + k;

        int n = max_kk;
        while (n > 0)
        {
            const size_t vl = vsetvl_i8m1(n);
            vint8m1_t _r0 = vle8_v_i8m1(p0, vl);
            vint8m1_t _r1 = vle8_v_i8m1(p1, vl);
            vint8m1_t _r2 = vle8_v_i8m1(p2, vl);
            vint8m1_t _r3 = vle8_v_i8m1(p3, vl);
            vint8m1_t _r4 = vle8_v_i8m1(p4, vl);
            vint8m1_t _r5 = vle8_v_i8m1(p5, vl);
            vint8m1_t _r6 = vle8_v_i8m1(p6, vl);
            vint8m1_t _r7 = vle8_v_i8m1(p7, vl);

            // transpose
            vsseg8e8_v_i8m1(pp, _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);

            pp += vl * 8;
            p0 += vl;
            p1 += vl;
            p2 += vl;
            p3 += vl;
            p4 += vl;
            p5 += vl;
            p6 += vl;
            p7 += vl;
            n -= vl;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;
        const signed char* p2 = (const signed char*)A + (i + ii + 2) * A_hstep + k;
        const signed char* p3 = (const signed char*)A + (i + ii + 3) * A_hstep + k;

        int n = max_kk;
        while (n > 0)
        {
            const size_t vl = vsetvl_i8m1(n);
            vint8m1_t _r0 = vle8_v_i8m1(p0, vl);
            vint8m1_t _r1 = vle8_v_i8m1(p1, vl);
            vint8m1_t _r2 = vle8_v_i8m1(p2, vl);
            vint8m1_t _r3 = vle8_v_i8m1(p3, vl);

            // transpose
            vsseg4e8_v_i8m1(pp, _r0, _r1, _r2, _r3, vl);

            pp += vl * 8;
            p0 += vl;
            p1 += vl;
            p2 += vl;
            p3 += vl;
            n -= vl;
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;

        int n = max_kk;
#if __riscv_vector
        while (n > 0)
        {
            const size_t vl = vsetvl_i8m1(n);
            vint8m1_t _r0 = vle8_v_i8m1(p0, vl);
            vint8m1_t _r1 = vle8_v_i8m1(p1, vl);

            // transpose
            vsseg2e8_v_i8m1(pp, _r0, _r1, vl);

            pp += vl * 8;
            p0 += vl;
            p1 += vl;
            n -= vl;
        }
#endif // __riscv_vector
        while (n > 0)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
            n--;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;

        int n = max_kk;
#if __riscv_vector
        while (n > 0)
        {
            const size_t vl = vsetvl_i8m1(n);
            vint8m1_t _r0 = vle8_v_i8m1(p0, vl);

            vse8_v_i8m1(pp, _r0, vl);

            pp += vl * 8;
            p0 += vl;
            n -= vl;
        }
#endif // __riscv_vector
        while (n > 0)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
            n--;
        }
    }
}

static void convolution_gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    // NCNN_LOGE("convolution_gemm_transB_packed_tile_int8 %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.cstep;

    const signed char* pAT = AT_tile;
    const signed char* pBT = BT_tile;

    int* outptr = topT_tile;

    int ii = 0;
#if __riscv_vector
    for (; ii + 7 < max_ii; ii += 8)
    {
        int* outptr0 = (int*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const signed char* pB = pBT;

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

            vint32m2_t _sum0;
            vint32m2_t _sum1;
            vint32m2_t _sum2;
            vint32m2_t _sum3;
            vint32m2_t _sum4;
            vint32m2_t _sum5;
            vint32m2_t _sum6;
            vint32m2_t _sum7;

            if (k == 0)
            {
                const int vl = 8;
                _sum0 = vmv_v_x_i32m2(0, vl);
                _sum1 = vmv_v_x_i32m2(0, vl);
                _sum2 = vmv_v_x_i32m2(0, vl);
                _sum3 = vmv_v_x_i32m2(0, vl);
                _sum4 = vmv_v_x_i32m2(0, vl);
                _sum5 = vmv_v_x_i32m2(0, vl);
                _sum6 = vmv_v_x_i32m2(0, vl);
                _sum7 = vmv_v_x_i32m2(0, vl);
            }
            else
            {
                const int vl = 8;
                _sum0 = vle32_v_i32m2(outptr, vl);
                _sum1 = vle32_v_i32m2(outptr + 8, vl);
                _sum2 = vle32_v_i32m2(outptr + 16, vl);
                _sum3 = vle32_v_i32m2(outptr + 24, vl);
                _sum4 = vle32_v_i32m2(outptr + 32, vl);
                _sum5 = vle32_v_i32m2(outptr + 40, vl);
                _sum6 = vle32_v_i32m2(outptr + 48, vl);
                _sum7 = vle32_v_i32m2(outptr + 56, vl);
            }

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                const int vl = 16;
                // A0: x0x1x2x3x4x5x6x7 x8x9xaxbxcxdxexf
                // A1: x4x5x6x7x0x1x2x3 xcxdxexfx8x9xaxb
                // A2: x2x3x0x1x6x7x4x5 xaxbx8x9xexfxcxd
                // A3: x6x7x4x5x2x3x0x1 xexfxcxdxaxbx8x9

                // B0: y0y1y2y3y4y5y6y7 y8y9yaybycydyeyf
                // B1: y3y2y1y0y7y6y5y4 ybyay9y8yfyeydyc
                uint8_t _pA1_perm_idx_arr[vl] = {4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11};
                uint8_t _pA2_perm_idx_arr[vl] = {2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13};
                uint8_t _pA3_perm_idx_arr[vl] = {6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9};
                uint8_t _pB1_perm_idx_arr[vl] = {3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12};
                vuint8m1_t _pA1_perm_idx = vle8_v_u8m1(_pA1_perm_idx_arr, vl);
                vuint8m1_t _pA2_perm_idx = vle8_v_u8m1(_pA2_perm_idx_arr, vl);
                vuint8m1_t _pA3_perm_idx = vle8_v_u8m1(_pA3_perm_idx_arr, vl);
                vuint8m1_t _pB1_perm_idx = vle8_v_u8m1(_pB1_perm_idx_arr, vl);

                vint8m1_t _pA0 = vle8_v_i8m1(pA, vl);
                vint8m1_t _pA1 = vrgather_vx_i8m1(_pA0, _pA1_perm_idx, vl);
                vint8m1_t _pA2 = vrgather_vx_i8m1(_pA0, _pA2_perm_idx, vl);
                vint8m1_t _pA3 = vrgather_vx_i8m1(_pA0, _pA3_perm_idx, vl);
                vint8m1_t _pB0 = vle8_v_i8m1(pB, vl);
                vint8m1_t _pB1 = vrgather_vx_i8m1(_pB0, _pB1_perm_idx, vl);

                vint16m2_t _s0 = vwmul_vv_i16m2(_pA0, _pB0, vl);
                vint16m2_t _s1 = vwmul_vv_i16m2(_pA1, _pB0, vl);
                vint16m2_t _s2 = vwmul_vv_i16m2(_pA2, _pB0, vl);
                vint16m2_t _s3 = vwmul_vv_i16m2(_pA3, _pB0, vl);
                vint16m2_t _s4 = vwmul_vv_i16m2(_pA0, _pB1, vl);
                vint16m2_t _s5 = vwmul_vv_i16m2(_pA1, _pB1, vl);
                vint16m2_t _s6 = vwmul_vv_i16m2(_pA2, _pB1, vl);
                vint16m2_t _s7 = vwmul_vv_i16m2(_pA3, _pB1, vl);

                _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(_s0, 0), vl);
                _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(_s1, 0), vl);
                _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(_s2, 0), vl);
                _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(_s3, 0), vl);
                _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(_s4, 0), vl);
                _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(_s5, 0), vl);
                _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(_s6, 0), vl);
                _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(_s7, 0), vl);
                _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(_s0, 1), vl);
                _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(_s1, 1), vl);
                _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(_s2, 1), vl);
                _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(_s3, 1), vl);
                _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(_s4, 1), vl);
                _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(_s5, 1), vl);
                _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(_s6, 1), vl);
                _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(_s7, 1), vl);

                pA += 16;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                const int vl = 16;
                // A0: x0x1x2x3x4x5x6x7 x4x5x6x7x0x1x2x3
                // A1: x2x3x0x1x6x7x4x5 x6x7x4x5x2x3x0x1

                // B0: y0y1y2y3y4y5y6y7 y0y1y2y3y4y5y6y7
                // B1: y3y2y1y0y7y6y5y4 y3y2y1y0y7y6y5y4
                uint8_t _pA0_perm_idx_arr[vl] = {0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 0, 1, 2, 3};
                uint8_t _pA1_perm_idx_arr[vl] = {2, 3, 0, 1, 6, 7, 4, 5, 6, 7, 4, 5, 2, 3, 0, 1};
                uint8_t _pB0_perm_idx_arr[vl] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
                uint8_t _pB1_perm_idx_arr[vl] = {3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4};
                vuint8m1_t _pA0_perm_idx = vle8_v_u8m1(_pA0_perm_idx_arr, vl);
                vuint8m1_t _pA1_perm_idx = vle8_v_u8m1(_pA1_perm_idx_arr, vl);
                vuint8m1_t _pB0_perm_idx = vle8_v_u8m1(_pB0_perm_idx_arr, vl);
                vuint8m1_t _pB1_perm_idx = vle8_v_u8m1(_pB1_perm_idx_arr, vl);

                vint8m1_t _pA0 = vle8_v_i8m1(pA, 8);
                vint8m1_t _pA0 = vrgather_vx_i8m1(_pA0, _pA0_perm_idx, vl);
                vint8m1_t _pA1 = vrgather_vx_i8m1(_pA0, _pA1_perm_idx, vl);
                vint8m1_t _pB0 = vle8_v_i8m1(pB, 8);
                vint8m1_t _pB0 = vrgather_vx_i8m1(_pB0, _pB0_perm_idx, vl);
                vint8m1_t _pB1 = vrgather_vx_i8m1(_pB0, _pB1_perm_idx, vl);

                vint16m2_t _s0 = vwmul_vv_i16m2(_pA0, _pB0, vl);
                vint16m2_t _s1 = vwmul_vv_i16m2(_pA1, _pB0, vl);
                vint16m2_t _s2 = vwmul_vv_i16m2(_pA0, _pB1, vl);
                vint16m2_t _s3 = vwmul_vv_i16m2(_pA1, _pB1, vl);

                _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(_s0, 0), vl);
                _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(_s0, 1), vl);
                _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(_s1, 0), vl);
                _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(_s1, 1), vl);
                _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(_s2, 0), vl);
                _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(_s2, 1), vl);
                _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(_s3, 0), vl);
                _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(_s3, 1), vl);

                pA += 8;
                pB += 8;
            }

            if (k_end)
            {
                // from
                //      abcdefgh efghabcd
                //      cdabghef ghefcdab
                //              x
                //      01234567 01234567
                //      32107654 32107654
                //              =
                //      a0 b1 c2 d3 e4 f5 g6 h7
                //      e0 f1 g2 h3 a4 b5 c6 d7
                //      c0 d1 a2 b3 g4 h5 e6 f7
                //      g0 h1 e2 f3 c4 d5 a6 b7
                //      a3 b2 c1 d0 e7 f6 g5 h4
                //      e3 f2 g1 h0 a7 b6 c5 d4
                //      c3 d2 a1 b0 g7 h6 e5 f4
                //      g3 h2 e1 f0 c7 d6 a5 b4
                if (out_elempack == 8)
                {
                    const int vl = 32;
                    // to
                    //      a0 b0 c0 d0
                    //      e0 f0 g0 h0
                    //      a1 b1 c1 d1
                    //      e1 f1 g1 h1
                    //      a2 b2 c2 d2
                    //      e2 f2 g2 h2
                    //      a3 b3 c3 d3
                    //      e3 f3 g3 h3
                    //      a4 b4 c4 d4
                    //      e4 f4 g4 h4
                    //      a5 b5 c5 d5
                    //      e5 f5 g5 h5
                    //      a6 b6 c6 d6
                    //      e6 f6 g6 h6
                    //      a7 b7 c7 d7
                    //      e7 f7 g7 h7
                    vint32m8_t _t0 = vundefined_i32m8();
                    vint32m8_t _t1 = vundefined_i32m8();
                    vset_v_i32m2_i32m8(_t0, 0, _sum0);
                    vset_v_i32m2_i32m8(_t0, 1, _sum6);
                    vset_v_i32m2_i32m8(_t0, 2, _sum2);
                    vset_v_i32m2_i32m8(_t0, 3, _sum4);
                    vset_v_i32m2_i32m8(_t1, 0, _sum1);
                    vset_v_i32m2_i32m8(_t1, 1, _sum7);
                    vset_v_i32m2_i32m8(_t1, 2, _sum3);
                    vset_v_i32m2_i32m8(_t1, 3, _sum5);
                    // after
                    //     00 01 02 03 04 05 06 07   08 09 0a 0b 0c 0d 0e 0f   10 11 12 13 14 15 16 17   18 19 1a 1b 1c 1d 1e 1f
                    //    -------------------------------------------------------------------------------------------------------
                    //     a0 b1 c2 d3 e4 f5 g6 h7 | c3 d2 a1 b0 g7 h6 e5 f4 | c0 d1 a2 b3 g4 h5 e6 f7 | a3 b2 c1 d0 e7 f6 g5 h4
                    //     e0 f1 g2 h3 a4 b5 c6 d7 | g3 h2 e1 f0 c7 d6 a5 b4 | g0 h1 e2 f3 c4 d5 a6 b7 | e3 f2 g1 h0 a7 b6 c5 d4
                    uint32_t _perm_idx_arr[vl] = {
                        0x00, 0x0b, 0x10, 0x1b, // a0 b0 c0 d0
                        0x0a, 0x01, 0x1a, 0x11, // a1 b1 c1 d1
                        0x12, 0x19, 0x02, 0x09, // a2 b2 c2 d2
                        0x18, 0x13, 0x08, 0x03, // a3 b3 c3 d3
                        0x04, 0x0f, 0x14, 0x1f, // e4 f4 g4 h4
                        0x0e, 0x05, 0x1e, 0x15, // e5 f5 g5 h5
                        0x16, 0x1d, 0x06, 0x0d, // e6 f6 g6 h6
                        0x1c, 0x17, 0x0c, 0x07, // e7 f7 g7 h7
                    };
                    vuint32m8_t _perm_idx = vle32_v_u32m8(_perm_idx_arr, vl);
                    _t0 = vrgather_vv_i32m8(_t0, _perm_idx, vl);
                    _t1 = vrgather_vv_i32m8(_t1, _perm_idx, vl);

                    vse32_v_i32m1(output0 + 0x00, vget_v_i32m8_i32m1(_t0, 0));
                    vse32_v_i32m1(output0 + 0x04, vget_v_i32m8_i32m1(_t1, 0));
                    vse32_v_i32m1(output0 + 0x08, vget_v_i32m8_i32m1(_t0, 1));
                    vse32_v_i32m1(output0 + 0x0c, vget_v_i32m8_i32m1(_t1, 1));
                    vse32_v_i32m1(output0 + 0x10, vget_v_i32m8_i32m1(_t0, 2));
                    vse32_v_i32m1(output0 + 0x14, vget_v_i32m8_i32m1(_t1, 2));
                    vse32_v_i32m1(output0 + 0x18, vget_v_i32m8_i32m1(_t0, 3));
                    vse32_v_i32m1(output0 + 0x1c, vget_v_i32m8_i32m1(_t1, 3));
                    vse32_v_i32m1(output0 + 0x20, vget_v_i32m8_i32m1(_t1, 4));
                    vse32_v_i32m1(output0 + 0x24, vget_v_i32m8_i32m1(_t0, 4));
                    vse32_v_i32m1(output0 + 0x28, vget_v_i32m8_i32m1(_t1, 5));
                    vse32_v_i32m1(output0 + 0x2c, vget_v_i32m8_i32m1(_t0, 5));
                    vse32_v_i32m1(output0 + 0x30, vget_v_i32m8_i32m1(_t1, 6));
                    vse32_v_i32m1(output0 + 0x34, vget_v_i32m8_i32m1(_t0, 6));
                    vse32_v_i32m1(output0 + 0x38, vget_v_i32m8_i32m1(_t1, 7));
                    vse32_v_i32m1(output0 + 0x3c, vget_v_i32m8_i32m1(_t0, 7));
                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
                    // to
                    //      a0 b0 c0 d0
                    //      a1 b1 c1 d1
                    //      a2 b2 c2 d2
                    //      a3 b3 c3 d3
                    //      a4 b4 c4 d4
                    //      a5 b5 c5 d5
                    //      a6 b6 c6 d6
                    //      a7 b7 c7 d7
                    //      e0 f0 g0 h0
                    //      e1 f1 g1 h1
                    //      e2 f2 g2 h2
                    //      e3 f3 g3 h3
                    //      e4 f4 g4 h4
                    //      e5 f5 g5 h5
                    //      e6 f6 g6 h6
                    //      e7 f7 g7 h7
                    vint32m8_t _t0 = vundefined_i32m8();
                    vint32m8_t _t1 = vundefined_i32m8();
                    vset_v_i32m2_i32m8(_t0, 0, _sum0);
                    vset_v_i32m2_i32m8(_t0, 1, _sum6);
                    vset_v_i32m2_i32m8(_t0, 2, _sum2);
                    vset_v_i32m2_i32m8(_t0, 3, _sum4);
                    vset_v_i32m2_i32m8(_t1, 0, _sum1);
                    vset_v_i32m2_i32m8(_t1, 1, _sum7);
                    vset_v_i32m2_i32m8(_t1, 2, _sum3);
                    vset_v_i32m2_i32m8(_t1, 3, _sum5);
                    // after
                    //     00 01 02 03 04 05 06 07   08 09 0a 0b 0c 0d 0e 0f   10 11 12 13 14 15 16 17   18 19 1a 1b 1c 1d 1e 1f
                    //    -------------------------------------------------------------------------------------------------------
                    //     a0 b1 c2 d3 e4 f5 g6 h7 | c3 d2 a1 b0 g7 h6 e5 f4 | c0 d1 a2 b3 g4 h5 e6 f7 | a3 b2 c1 d0 e7 f6 g5 h4
                    //     e0 f1 g2 h3 a4 b5 c6 d7 | g3 h2 e1 f0 c7 d6 a5 b4 | g0 h1 e2 f3 c4 d5 a6 b7 | e3 f2 g1 h0 a7 b6 c5 d4
                    uint32_t _perm_idx_arr[vl] = {
                        0x00, 0x0b, 0x10, 0x1b, // a0 b0 c0 d0
                        0x0a, 0x01, 0x1a, 0x11, // a1 b1 c1 d1
                        0x12, 0x19, 0x02, 0x09, // a2 b2 c2 d2
                        0x18, 0x13, 0x08, 0x03, // a3 b3 c3 d3
                        0x04, 0x0f, 0x14, 0x1f, // e4 f4 g4 h4
                        0x0e, 0x05, 0x1e, 0x15, // e5 f5 g5 h5
                        0x16, 0x1d, 0x06, 0x0d, // e6 f6 g6 h6
                        0x1c, 0x17, 0x0c, 0x07, // e7 f7 g7 h7
                    };
                    vuint32m8_t _perm_idx = vle32_v_u32m8(_perm_idx_arr, vl);
                    _t0 = vrgather_vv_i32m8(_t0, _perm_idx, vl);
                    _t1 = vrgather_vv_i32m8(_t1, _perm_idx, vl);

                    vse32_v_i32m1(output0 + 0x00, vget_v_i32m8_i32m1(_t0, 0));
                    vse32_v_i32m1(output0 + 0x04, vget_v_i32m8_i32m1(_t0, 1));
                    vse32_v_i32m1(output0 + 0x08, vget_v_i32m8_i32m1(_t0, 2));
                    vse32_v_i32m1(output0 + 0x0c, vget_v_i32m8_i32m1(_t0, 3));
                    vse32_v_i32m1(output0 + 0x10, vget_v_i32m8_i32m1(_t1, 4));
                    vse32_v_i32m1(output0 + 0x14, vget_v_i32m8_i32m1(_t1, 5));
                    vse32_v_i32m1(output0 + 0x18, vget_v_i32m8_i32m1(_t1, 6));
                    vse32_v_i32m1(output0 + 0x1c, vget_v_i32m8_i32m1(_t1, 7));

                    vse32_v_i32m1(output0 + out_hstep * 4 + 0x00, vget_v_i32m8_i32m1(_t1, 0));
                    vse32_v_i32m1(output0 + out_hstep * 4 + 0x04, vget_v_i32m8_i32m1(_t1, 1));
                    vse32_v_i32m1(output0 + out_hstep * 4 + 0x08, vget_v_i32m8_i32m1(_t1, 2));
                    vse32_v_i32m1(output0 + out_hstep * 4 + 0x0c, vget_v_i32m8_i32m1(_t1, 3));
                    vse32_v_i32m1(output0 + out_hstep * 4 + 0x10, vget_v_i32m8_i32m1(_t0, 4));
                    vse32_v_i32m1(output0 + out_hstep * 4 + 0x14, vget_v_i32m8_i32m1(_t0, 5));
                    vse32_v_i32m1(output0 + out_hstep * 4 + 0x18, vget_v_i32m8_i32m1(_t0, 6));
                    vse32_v_i32m1(output0 + out_hstep * 4 + 0x1c, vget_v_i32m8_i32m1(_t0, 7));
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 a2 a3
                    //      a4 a5 a6 a7
                    //      b0 b1 b2 b3
                    //      b4 b5 b6 b7
                    //      c0 c1 c2 c3
                    //      c4 c5 c6 c7
                    //      d0 d1 d2 d3
                    //      d4 d5 d6 d7
                    //      e0 e1 e2 e3
                    //      e4 e5 e6 e7
                    //      f0 f1 f2 f3
                    //      f4 f5 f6 f7
                    //      g0 g1 g2 g3
                    //      g4 g5 g6 g7
                    //      h0 h1 h2 h3
                    //      h4 h5 h6 h7
                    vint32m8_t _t0 = vundefined_i32m8();
                    vint32m8_t _t1 = vundefined_i32m8();
                    vset_v_i32m2_i32m8(_t0, 0, _sum0);
                    vset_v_i32m2_i32m8(_t0, 1, _sum6);
                    vset_v_i32m2_i32m8(_t0, 2, _sum2);
                    vset_v_i32m2_i32m8(_t0, 3, _sum4);
                    vset_v_i32m2_i32m8(_t1, 0, _sum1);
                    vset_v_i32m2_i32m8(_t1, 1, _sum7);
                    vset_v_i32m2_i32m8(_t1, 2, _sum3);
                    vset_v_i32m2_i32m8(_t1, 3, _sum5);
                    // after
                    //     00 01 02 03 04 05 06 07   08 09 0a 0b 0c 0d 0e 0f   10 11 12 13 14 15 16 17   18 19 1a 1b 1c 1d 1e 1f
                    //    -------------------------------------------------------------------------------------------------------
                    //     a0 b1 c2 d3 e4 f5 g6 h7 | c3 d2 a1 b0 g7 h6 e5 f4 | c0 d1 a2 b3 g4 h5 e6 f7 | a3 b2 c1 d0 e7 f6 g5 h4
                    //     e0 f1 g2 h3 a4 b5 c6 d7 | g3 h2 e1 f0 c7 d6 a5 b4 | g0 h1 e2 f3 c4 d5 a6 b7 | e3 f2 g1 h0 a7 b6 c5 d4
                    uint32_t _perm_idx_arr[vl] = {
                        0x00, 0x0a, 0x12, 0x18, // a0 a1 a2 a3
                        0x0b, 0x01, 0x19, 0x13, // b0 b1 b2 b3
                        0x10, 0x1a, 0x02, 0x08, // c0 c1 c2 c3
                        0x1b, 0x11, 0x09, 0x03, // d0 d1 d2 d3
                        0x04, 0x0e, 0x16, 0x1c, // e4 e5 e6 e7
                        0x0f, 0x05, 0x1d, 0x17, // f4 f5 f6 f7
                        0x14, 0x1e, 0x06, 0x0c, // g4 g5 g6 g7
                        0x1f, 0x15, 0x0d, 0x07, // h4 h5 h6 h7
                    } vuint32m8_t _perm_idx
                        = vle32_v_u32m8(_perm_idx_arr, vl);
                    _t0 = vrgather_vv_i32m8(_t0, _perm_idx, vl);
                    _t1 = vrgather_vv_i32m8(_t1, _perm_idx, vl);

                    vse32_v_i32m1(output0 + out_hstep * 0 + 0, vget_v_i32m8_i32m1(_t0, 0));
                    vse32_v_i32m1(output0 + out_hstep * 0 + 4, vget_v_i32m8_i32m1(_t1, 4));
                    vse32_v_i32m1(output0 + out_hstep * 1 + 0, vget_v_i32m8_i32m1(_t0, 1));
                    vse32_v_i32m1(output0 + out_hstep * 1 + 4, vget_v_i32m8_i32m1(_t1, 5));
                    vse32_v_i32m1(output0 + out_hstep * 2 + 0, vget_v_i32m8_i32m1(_t0, 2));
                    vse32_v_i32m1(output0 + out_hstep * 2 + 4, vget_v_i32m8_i32m1(_t1, 6));
                    vse32_v_i32m1(output0 + out_hstep * 3 + 0, vget_v_i32m8_i32m1(_t0, 4));
                    vse32_v_i32m1(output0 + out_hstep * 3 + 4, vget_v_i32m8_i32m1(_t1, 7));

                    vse32_v_i32m1(output0 + out_hstep * 4 + 0, vget_v_i32m8_i32m1(_t1, 0));
                    vse32_v_i32m1(output0 + out_hstep * 4 + 4, vget_v_i32m8_i32m1(_t0, 4));
                    vse32_v_i32m1(output0 + out_hstep * 5 + 0, vget_v_i32m8_i32m1(_t1, 1));
                    vse32_v_i32m1(output0 + out_hstep * 5 + 4, vget_v_i32m8_i32m1(_t0, 5));
                    vse32_v_i32m1(output0 + out_hstep * 6 + 0, vget_v_i32m8_i32m1(_t1, 2));
                    vse32_v_i32m1(output0 + out_hstep * 6 + 4, vget_v_i32m8_i32m1(_t0, 6));
                    vse32_v_i32m1(output0 + out_hstep * 7 + 0, vget_v_i32m8_i32m1(_t1, 4));
                    vse32_v_i32m1(output0 + out_hstep * 7 + 4, vget_v_i32m8_i32m1(_t0, 7));
                    outptr0 += 8;
                }
            }
            else
            {
                vse32_v_i32m2(outptr + 0x00, _sum0);
                vse32_v_i32m2(outptr + 0x10, _sum1);
                vse32_v_i32m2(outptr + 0x20, _sum2);
                vse32_v_i32m2(outptr + 0x30, _sum3);
                vse32_v_i32m2(outptr + 0x40, _sum4);
                vse32_v_i32m2(outptr + 0x50, _sum5);
                vse32_v_i32m2(outptr + 0x60, _sum6);
                vse32_v_i32m2(outptr + 0x70, _sum7);
            }

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            int32x4_t _sum0;
            int32x4_t _sum1;
            int32x4_t _sum2;
            int32x4_t _sum3;
            int32x4_t _sum4;
            int32x4_t _sum5;
            int32x4_t _sum6;
            int32x4_t _sum7;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
                _sum4 = vdupq_n_s32(0);
                _sum5 = vdupq_n_s32(0);
                _sum6 = vdupq_n_s32(0);
                _sum7 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
                _sum2 = vld1q_s32(outptr + 8);
                _sum3 = vld1q_s32(outptr + 12);
                _sum4 = vld1q_s32(outptr + 16);
                _sum5 = vld1q_s32(outptr + 20);
                _sum6 = vld1q_s32(outptr + 24);
                _sum7 = vld1q_s32(outptr + 28);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA2 = vld1q_s8(pA + 16);
                int8x16_t _pB02 = vld1q_s8(pB);

                // aabbccdd eeffgghh

                // ccddaabb gghheeff

                int8x16_t _pA1 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA0)));
                int8x16_t _pA3 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA2)));

                // 00112233 44556677

                // 33221100 77665544

                int8x16_t _pB13 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB02)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB02));
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB02));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB02));
                int16x8_t _s3 = vmull_s8(vget_high_s8(_pA1), vget_low_s8(_pB02));
                int16x8_t _s4 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB13));
                int16x8_t _s5 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB13));
                int16x8_t _s6 = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB13));
                int16x8_t _s7 = vmull_s8(vget_high_s8(_pA1), vget_low_s8(_pB13));

                _s0 = vmlal_s8(_s0, vget_low_s8(_pA2), vget_high_s8(_pB02));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA2), vget_high_s8(_pB02));
                _s2 = vmlal_s8(_s2, vget_low_s8(_pA3), vget_high_s8(_pB02));
                _s3 = vmlal_s8(_s3, vget_high_s8(_pA3), vget_high_s8(_pB02));
                _s4 = vmlal_s8(_s4, vget_low_s8(_pA2), vget_high_s8(_pB13));
                _s5 = vmlal_s8(_s5, vget_high_s8(_pA2), vget_high_s8(_pB13));
                _s6 = vmlal_s8(_s6, vget_low_s8(_pA3), vget_high_s8(_pB13));
                _s7 = vmlal_s8(_s7, vget_high_s8(_pA3), vget_high_s8(_pB13));

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);

                pA += 32;
                pB += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x8_t _pB0 = vld1_s8(pB);

                // aabbccdd eeffgghh

                // ccddaabb gghheeff

                int8x16_t _pA1 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA0)));

                // 00112233

                // 33221100

                int8x8_t _pB1 = vreinterpret_s8_s16(vrev64_s16(vreinterpret_s16_s8(_pB0)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), _pB0);
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), _pB0);
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA1), _pB0);
                int16x8_t _s3 = vmull_s8(vget_high_s8(_pA1), _pB0);
                int16x8_t _s4 = vmull_s8(vget_low_s8(_pA0), _pB1);
                int16x8_t _s5 = vmull_s8(vget_high_s8(_pA0), _pB1);
                int16x8_t _s6 = vmull_s8(vget_low_s8(_pA1), _pB1);
                int16x8_t _s7 = vmull_s8(vget_high_s8(_pA1), _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);

                pA += 16;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA0 = vld1_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));
                // int8x8_t _pB0 = vld1_s8(pB);
                // _pB0 = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_pB0), vreinterpret_s32_s8(_pB0)).val[0]);

                // abcdefgh  ->  cdabghef
                int8x8_t _pA1 = vreinterpret_s8_s16(vrev32_s16(vreinterpret_s16_s8(_pA0)));

                // 01230123  ->  32103210
                int8x8_t _pB1 = vrev64_s8(_pB0);

                int16x8_t _s01 = vmull_s8(_pA0, _pB0);
                int16x8_t _s23 = vmull_s8(_pA1, _pB0);
                int16x8_t _s45 = vmull_s8(_pA0, _pB1);
                int16x8_t _s67 = vmull_s8(_pA1, _pB1);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s01));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s23));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s23));
                _sum4 = vaddw_s16(_sum4, vget_low_s16(_s45));
                _sum5 = vaddw_s16(_sum5, vget_high_s16(_s45));
                _sum6 = vaddw_s16(_sum6, vget_low_s16(_s67));
                _sum7 = vaddw_s16(_sum7, vget_high_s16(_s67));

                pA += 8;
                pB += 4;
            }

            if (k_end)
            {
                // from
                //      a0 b1 c2 d3
                //      e0 f1 g2 h3
                //      c0 d1 a2 b3
                //      g0 h1 e2 f3
                //      a3 b2 c1 d0
                //      e3 f2 g1 h0
                //      c3 d2 a1 b0
                //      g3 h2 e1 f0
                if (out_elempack == 8)
                {
                    // to
                    //      a0 b0 c0 d0
                    //      e0 f0 g0 h0
                    //      a1 b1 c1 d1
                    //      e1 f1 g1 h1
                    //      a2 b2 c2 d2
                    //      e2 f2 g2 h2
                    //      a3 b3 c3 d3
                    //      e3 f3 g3 h3
                    {
                        _sum4 = vrev64q_s32(_sum4);
                        _sum5 = vrev64q_s32(_sum5);
                        _sum6 = vrev64q_s32(_sum6);
                        _sum7 = vrev64q_s32(_sum7);
                        _sum4 = vextq_s32(_sum4, _sum4, 2);
                        _sum5 = vextq_s32(_sum5, _sum5, 2);
                        _sum6 = vextq_s32(_sum6, _sum6, 2);
                        _sum7 = vextq_s32(_sum7, _sum7, 2);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                        int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                        int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                        int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                        _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                        _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                        _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                        _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                        _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                        _sum2 = vrev64q_s32(_sum2);
                        _sum3 = vrev64q_s32(_sum3);
                        _sum6 = vrev64q_s32(_sum6);
                        _sum7 = vrev64q_s32(_sum7);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);
                    vst1q_s32(outptr0 + 16, _sum4);
                    vst1q_s32(outptr0 + 20, _sum5);
                    vst1q_s32(outptr0 + 24, _sum6);
                    vst1q_s32(outptr0 + 28, _sum7);
                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    // to
                    //      a0 b0 c0 d0
                    //      a1 b1 c1 d1
                    //      a2 b2 c2 d2
                    //      a3 b3 c3 d3
                    //      e0 f0 g0 h0
                    //      e1 f1 g1 h1
                    //      e2 f2 g2 h2
                    //      e3 f3 g3 h3
                    {
                        _sum4 = vrev64q_s32(_sum4);
                        _sum5 = vrev64q_s32(_sum5);
                        _sum6 = vrev64q_s32(_sum6);
                        _sum7 = vrev64q_s32(_sum7);
                        _sum4 = vextq_s32(_sum4, _sum4, 2);
                        _sum5 = vextq_s32(_sum5, _sum5, 2);
                        _sum6 = vextq_s32(_sum6, _sum6, 2);
                        _sum7 = vextq_s32(_sum7, _sum7, 2);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                        int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                        int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                        int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                        _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                        _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                        _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                        _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                        _sum3 = vrev64q_s32(_sum3);
                        _sum5 = vrev64q_s32(_sum5);
                        _sum7 = vrev64q_s32(_sum7);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);
                    vst1q_s32(outptr0 + out_hstep * 4, _sum4);
                    vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum5);
                    vst1q_s32(outptr0 + out_hstep * 4 + 8, _sum6);
                    vst1q_s32(outptr0 + out_hstep * 4 + 12, _sum7);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 a2 a3
                    //      b0 b1 b2 b3
                    //      c0 c1 c2 c3
                    //      d0 d1 d2 d3
                    //      e0 e1 e2 e3
                    //      f0 f1 f2 f3
                    //      g0 g1 g2 g3
                    //      h0 h1 h2 h3
                    {
                        _sum2 = vextq_s32(_sum2, _sum2, 2);
                        _sum3 = vextq_s32(_sum3, _sum3, 2);
                        _sum6 = vextq_s32(_sum6, _sum6, 2);
                        _sum7 = vextq_s32(_sum7, _sum7, 2);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                        int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                        int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                        int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                        _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                        _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                        _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                        _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                        _sum3 = vrev64q_s32(_sum3);
                        _sum5 = vrev64q_s32(_sum5);
                        _sum7 = vrev64q_s32(_sum7);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + out_hstep, _sum1);
                    vst1q_s32(outptr0 + out_hstep * 2, _sum2);
                    vst1q_s32(outptr0 + out_hstep * 3, _sum3);
                    vst1q_s32(outptr0 + out_hstep * 4, _sum4);
                    vst1q_s32(outptr0 + out_hstep * 5, _sum5);
                    vst1q_s32(outptr0 + out_hstep * 6, _sum6);
                    vst1q_s32(outptr0 + out_hstep * 7, _sum7);
                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
                vst1q_s32(outptr + 16, _sum4);
                vst1q_s32(outptr + 20, _sum5);
                vst1q_s32(outptr + 24, _sum6);
                vst1q_s32(outptr + 28, _sum7);
            }

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            int32x4_t _sum0;
            int32x4_t _sum1;
            int32x4_t _sum2;
            int32x4_t _sum3;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
                _sum2 = vld1q_s32(outptr + 8);
                _sum3 = vld1q_s32(outptr + 12);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA2 = vld1q_s8(pA + 16);
                int8x8_t _pB = vld1_s8(pB);

                // aabbccdd eeffgghh   aabbccdd eeffgghh

                // 00112233 -> 00110011 22332233

                // 11001100 33223322

                int32x2x2_t _pBB = vzip_s32(vreinterpret_s32_s8(_pB), vreinterpret_s32_s8(_pB));
                int8x16_t _pB02 = vreinterpretq_s8_s32(vcombine_s32(_pBB.val[0], _pBB.val[1]));

                int8x16_t _pB13 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB02)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB02));
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB02));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB13));
                int16x8_t _s3 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB13));
                _s0 = vmlal_s8(_s0, vget_low_s8(_pA2), vget_high_s8(_pB02));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA2), vget_high_s8(_pB02));
                _s2 = vmlal_s8(_s2, vget_low_s8(_pA2), vget_high_s8(_pB13));
                _s3 = vmlal_s8(_s3, vget_high_s8(_pA2), vget_high_s8(_pB13));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);

                pA += 32;
                pB += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));

                // aabbccdd eeffgghh

                // 00110011
                // 11001100

                int8x8_t _pB1 = vreinterpret_s8_s16(vrev64_s16(vreinterpret_s16_s8(_pB0)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), _pB0);
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA), _pB0);
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA), _pB1);
                int16x8_t _s3 = vmull_s8(vget_high_s8(_pA), _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);

                pA += 16;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));

                // abcdefgh

                // 01010101
                // 10101010
                int8x8_t _pB1 = vext_s8(_pB0, _pB0, 1);

                int16x8_t _s0 = vmull_s8(_pA, _pB0);
                int16x8_t _s1 = vmull_s8(_pA, _pB1);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));

                pA += 8;
                pB += 2;
            }

            if (k_end)
            {
                // from
                //      a0 b1 c0 d1
                //      e0 f1 g0 h1
                //      a1 b0 c1 d0
                //      e1 f0 g1 h0
                if (out_elempack == 8)
                {
                    // to
                    //      a0 b0 c0 d0
                    //      e0 f0 g0 h0
                    //      a1 b1 c1 d1
                    //      e1 f1 g1 h1
                    {
                        _sum2 = vrev64q_s32(_sum2);
                        _sum3 = vrev64q_s32(_sum3);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum2);
                        int32x4x2_t _t1 = vzipq_s32(_sum1, _sum3);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                        _sum1 = vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t1.val[1]));
                        _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t1.val[1]));
                        _sum2 = vrev64q_s32(_sum2);
                        _sum3 = vrev64q_s32(_sum3);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);
                    outptr0 += 16;
                }
                if (out_elempack == 4)
                {
                    // to
                    //      a0 b0 c0 d0
                    //      a1 b1 c1 d1
                    //      e0 f0 g0 h0
                    //      e1 f1 g1 h1
                    {
                        _sum2 = vrev64q_s32(_sum2);
                        _sum3 = vrev64q_s32(_sum3);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum2);
                        int32x4x2_t _t1 = vzipq_s32(_sum1, _sum3);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                        _sum2 = vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t1.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t1.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                        _sum3 = vrev64q_s32(_sum3);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + out_hstep * 4, _sum2);
                    vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum3);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 c0 c1
                    //      b0 b1 d0 d1
                    //      e0 e1 g0 g1
                    //      f0 f1 h0 h1
                    {
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum2);
                        int32x4x2_t _t1 = vzipq_s32(_sum1, _sum3);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                        _sum2 = vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t1.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t1.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                        _sum3 = vrev64q_s32(_sum3);
                    }

                    vst1_s32(outptr0, vget_low_s32(_sum0));
                    vst1_s32(outptr0 + out_hstep, vget_low_s32(_sum1));
                    vst1_s32(outptr0 + out_hstep * 2, vget_high_s32(_sum0));
                    vst1_s32(outptr0 + out_hstep * 3, vget_high_s32(_sum1));
                    vst1_s32(outptr0 + out_hstep * 4, vget_low_s32(_sum2));
                    vst1_s32(outptr0 + out_hstep * 5, vget_low_s32(_sum3));
                    vst1_s32(outptr0 + out_hstep * 6, vget_high_s32(_sum2));
                    vst1_s32(outptr0 + out_hstep * 7, vget_high_s32(_sum3));
                    outptr0 += 2;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
            }

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            int32x4_t _sum0;
            int32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA2 = vld1q_s8(pA + 16);
                int8x8_t _pB0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));
                int8x8_t _pB1 = vreinterpret_s8_s16(vld1_dup_s16((const short*)(pB + 2)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), _pB0);
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), _pB0);
                _s0 = vmlal_s8(_s0, vget_low_s8(_pA2), _pB1);
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA2), _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);

                pA += 32;
                pB += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), _pB);
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA), _pB);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);

                pA += 16;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vld1_dup_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
                if (out_elempack == 4)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + out_hstep * 4, _sum1);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    outptr0[0] = vgetq_lane_s32(_sum0, 0);
                    outptr0[out_hstep] = vgetq_lane_s32(_sum0, 1);
                    outptr0[out_hstep * 2] = vgetq_lane_s32(_sum0, 2);
                    outptr0[out_hstep * 3] = vgetq_lane_s32(_sum0, 3);
                    outptr0[out_hstep * 4] = vgetq_lane_s32(_sum1, 0);
                    outptr0[out_hstep * 5] = vgetq_lane_s32(_sum1, 1);
                    outptr0[out_hstep * 6] = vgetq_lane_s32(_sum1, 2);
                    outptr0[out_hstep * 7] = vgetq_lane_s32(_sum1, 3);
                    outptr0++;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
            }

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        int* outptr0 = (int*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const signed char* pB = pBT;

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

            int32x4_t _sum0;
            int32x4_t _sum1;
            int32x4_t _sum2;
            int32x4_t _sum3;
            int32x4_t _sum4;
            int32x4_t _sum5;
            int32x4_t _sum6;
            int32x4_t _sum7;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
                _sum4 = vdupq_n_s32(0);
                _sum5 = vdupq_n_s32(0);
                _sum6 = vdupq_n_s32(0);
                _sum7 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
                _sum2 = vld1q_s32(outptr + 8);
                _sum3 = vld1q_s32(outptr + 12);
                _sum4 = vld1q_s32(outptr + 16);
                _sum5 = vld1q_s32(outptr + 20);
                _sum6 = vld1q_s32(outptr + 24);
                _sum7 = vld1q_s32(outptr + 28);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x16_t _pA02 = vld1q_s8(pA);
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB2 = vld1q_s8(pB + 16);

                int8x16_t _pA13 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA02)));

                int8x16_t _pB1 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB0)));
                int8x16_t _pB3 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB2)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA02), vget_low_s8(_pB0));
                int16x8_t _s1 = vmull_s8(vget_low_s8(_pA02), vget_high_s8(_pB0));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA13), vget_low_s8(_pB0));
                int16x8_t _s3 = vmull_s8(vget_low_s8(_pA13), vget_high_s8(_pB0));
                int16x8_t _s4 = vmull_s8(vget_low_s8(_pA02), vget_low_s8(_pB1));
                int16x8_t _s5 = vmull_s8(vget_low_s8(_pA02), vget_high_s8(_pB1));
                int16x8_t _s6 = vmull_s8(vget_low_s8(_pA13), vget_low_s8(_pB1));
                int16x8_t _s7 = vmull_s8(vget_low_s8(_pA13), vget_high_s8(_pB1));

                _s0 = vmlal_s8(_s0, vget_high_s8(_pA02), vget_low_s8(_pB2));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA02), vget_high_s8(_pB2));
                _s2 = vmlal_s8(_s2, vget_high_s8(_pA13), vget_low_s8(_pB2));
                _s3 = vmlal_s8(_s3, vget_high_s8(_pA13), vget_high_s8(_pB2));
                _s4 = vmlal_s8(_s4, vget_high_s8(_pA02), vget_low_s8(_pB3));
                _s5 = vmlal_s8(_s5, vget_high_s8(_pA02), vget_high_s8(_pB3));
                _s6 = vmlal_s8(_s6, vget_high_s8(_pA13), vget_low_s8(_pB3));
                _s7 = vmlal_s8(_s7, vget_high_s8(_pA13), vget_high_s8(_pB3));

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);

                pA += 16;
                pB += 32;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8_t _pA0 = vld1_s8(pA);
                int8x16_t _pB0 = vld1q_s8(pB);

                // aabbccdd
                // ccddaabb

                int8x8_t _pA1 = vreinterpret_s8_s32(vrev64_s32(vreinterpret_s32_s8(_pA0)));

                // 00112233 44556677
                // 33221100 77665544

                int8x16_t _pB1 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB0)));

                int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                int16x8_t _s1 = vmull_s8(_pA0, vget_high_s8(_pB0));
                int16x8_t _s2 = vmull_s8(_pA1, vget_low_s8(_pB0));
                int16x8_t _s3 = vmull_s8(_pA1, vget_high_s8(_pB0));
                int16x8_t _s4 = vmull_s8(_pA0, vget_low_s8(_pB1));
                int16x8_t _s5 = vmull_s8(_pA0, vget_high_s8(_pB1));
                int16x8_t _s6 = vmull_s8(_pA1, vget_low_s8(_pB1));
                int16x8_t _s7 = vmull_s8(_pA1, vget_high_s8(_pB1));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);

                pA += 8;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                int8x8_t _pB0 = vld1_s8(pB);

                // abcd abcd
                // cdab cdab

                int8x8_t _pA1 = vext_s8(_pA0, _pA0, 2);

                // 0123 4567
                // 3210 7654

                int8x8_t _pB1 = vrev32_s8(_pB0);

                int16x8_t _s01 = vmull_s8(_pA0, _pB0);
                int16x8_t _s23 = vmull_s8(_pA1, _pB0);
                int16x8_t _s45 = vmull_s8(_pA0, _pB1);
                int16x8_t _s67 = vmull_s8(_pA1, _pB1);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s01));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s23));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s23));
                _sum4 = vaddw_s16(_sum4, vget_low_s16(_s45));
                _sum5 = vaddw_s16(_sum5, vget_high_s16(_s45));
                _sum6 = vaddw_s16(_sum6, vget_low_s16(_s67));
                _sum7 = vaddw_s16(_sum7, vget_high_s16(_s67));

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                // from
                //      a0 b1 c2 d3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      c7 d6 a5 b4
                if (out_elempack == 4)
                {
                    // to
                    //      a0 b0 c0 d0
                    //      a1 b1 c1 d1
                    //      a2 b2 c2 d2
                    //      a3 b3 c3 d3
                    //      a4 b4 c4 d4
                    //      a5 b5 c5 d5
                    //      a6 b6 c6 d6
                    //      a7 b7 c7 d7
                    {
                        _sum4 = vrev64q_s32(_sum4);
                        _sum5 = vrev64q_s32(_sum5);
                        _sum6 = vrev64q_s32(_sum6);
                        _sum7 = vrev64q_s32(_sum7);
                        _sum4 = vextq_s32(_sum4, _sum4, 2);
                        _sum5 = vextq_s32(_sum5, _sum5, 2);
                        _sum6 = vextq_s32(_sum6, _sum6, 2);
                        _sum7 = vextq_s32(_sum7, _sum7, 2);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                        int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                        int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                        int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                        _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                        _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                        _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                        _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                        _sum3 = vrev64q_s32(_sum3);
                        _sum5 = vrev64q_s32(_sum5);
                        _sum7 = vrev64q_s32(_sum7);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);
                    vst1q_s32(outptr0 + 16, _sum4);
                    vst1q_s32(outptr0 + 20, _sum5);
                    vst1q_s32(outptr0 + 24, _sum6);
                    vst1q_s32(outptr0 + 28, _sum7);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 a2 a3
                    //      a4 a5 a6 a7
                    //      b0 b1 b2 b3
                    //      b4 b5 b6 b7
                    //      c0 c1 c2 c3
                    //      c4 c5 c6 c7
                    //      d0 d1 d2 d3
                    //      d4 d5 d6 d7
                    {
                        _sum2 = vextq_s32(_sum2, _sum2, 2);
                        _sum3 = vextq_s32(_sum3, _sum3, 2);
                        _sum6 = vextq_s32(_sum6, _sum6, 2);
                        _sum7 = vextq_s32(_sum7, _sum7, 2);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                        int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                        int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                        int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                        _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                        _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                        _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                        _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                        _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                        _sum2 = vrev64q_s32(_sum2);
                        _sum3 = vrev64q_s32(_sum3);
                        _sum6 = vrev64q_s32(_sum6);
                        _sum7 = vrev64q_s32(_sum7);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + out_hstep, _sum2);
                    vst1q_s32(outptr0 + out_hstep + 4, _sum3);
                    vst1q_s32(outptr0 + out_hstep * 2, _sum4);
                    vst1q_s32(outptr0 + out_hstep * 2 + 4, _sum5);
                    vst1q_s32(outptr0 + out_hstep * 3, _sum6);
                    vst1q_s32(outptr0 + out_hstep * 3 + 4, _sum7);
                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
                vst1q_s32(outptr + 16, _sum4);
                vst1q_s32(outptr + 20, _sum5);
                vst1q_s32(outptr + 24, _sum6);
                vst1q_s32(outptr + 28, _sum7);
            }

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            int32x4_t _sum0;
            int32x4_t _sum1;
            int32x4_t _sum2;
            int32x4_t _sum3;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
                _sum2 = vld1q_s32(outptr + 8);
                _sum3 = vld1q_s32(outptr + 12);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x16_t _pA02 = vld1q_s8(pA);
                int8x16_t _pB02 = vld1q_s8(pB);

                // aabbccdd eeffgghh
                // ccddaabb gghheeff

                int8x16_t _pA13 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA02)));

                // 00112233 44556677
                // 33221100 77665544

                int8x16_t _pB13 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB02)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA02), vget_low_s8(_pB02));
                int16x8_t _s1 = vmull_s8(vget_low_s8(_pA13), vget_low_s8(_pB02));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA02), vget_low_s8(_pB13));
                int16x8_t _s3 = vmull_s8(vget_low_s8(_pA13), vget_low_s8(_pB13));

                _s0 = vmlal_s8(_s0, vget_high_s8(_pA02), vget_high_s8(_pB02));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA13), vget_high_s8(_pB02));
                _s2 = vmlal_s8(_s2, vget_high_s8(_pA02), vget_high_s8(_pB13));
                _s3 = vmlal_s8(_s3, vget_high_s8(_pA13), vget_high_s8(_pB13));

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);

                pA += 16;
                pB += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8_t _pA0 = vld1_s8(pA);
                int8x8_t _pB0 = vld1_s8(pB);

                // aabbccdd
                // ccddaabb

                int8x8_t _pA1 = vext_s8(_pA0, _pA0, 4);

                // 00112233
                // 33221100

                int8x8_t _pB1 = vreinterpret_s8_s16(vrev64_s16(vreinterpret_s16_s8(_pB0)));

                int16x8_t _s0 = vmull_s8(_pA0, _pB0);
                int16x8_t _s1 = vmull_s8(_pA1, _pB0);
                int16x8_t _s2 = vmull_s8(_pA0, _pB1);
                int16x8_t _s3 = vmull_s8(_pA1, _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);

                pA += 8;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA0 = vld1_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));

                // abcd.... -> cdab.... -> abcdcdab
                int8x8_t _pA1 = vreinterpret_s8_s16(vrev32_s16(vreinterpret_s16_s8(_pA0)));
                int8x8_t _pA01 = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_pA0), vreinterpret_s32_s8(_pA1)).val[0]);

                // 01230123 -> 32103210
                int8x8_t _pB1 = vrev32_s8(_pB0);

                int16x8_t _s01 = vmull_s8(_pA01, _pB0);
                int16x8_t _s23 = vmull_s8(_pA01, _pB1);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s01));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s23));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s23));

                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                // from
                //      a0 b1 c2 d3
                //      c0 d1 a2 b3
                //      a3 b2 c1 d0
                //      c3 d2 a1 b0
                if (out_elempack == 4)
                {
                    // to
                    //      a0 b0 c0 d0
                    //      a1 b1 c1 d1
                    //      a2 b2 c2 d2
                    //      a3 b3 c3 d3
                    {
                        _sum2 = vrev64q_s32(_sum2);
                        _sum3 = vrev64q_s32(_sum3);
                        _sum2 = vextq_s32(_sum2, _sum2, 2);
                        _sum3 = vextq_s32(_sum3, _sum3, 2);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum3);
                        int32x4x2_t _t1 = vzipq_s32(_sum1, _sum2);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                        _sum3 = vrev64q_s32(_sum3);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 a2 a3
                    //      b0 b1 b2 b3
                    //      c0 c1 c2 c3
                    //      d0 d1 d2 d3
                    {
                        _sum1 = vextq_s32(_sum1, _sum1, 2);
                        _sum3 = vextq_s32(_sum3, _sum3, 2);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum3);
                        int32x4x2_t _t1 = vzipq_s32(_sum1, _sum2);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                        _sum3 = vrev64q_s32(_sum3);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + out_hstep, _sum1);
                    vst1q_s32(outptr0 + out_hstep * 2, _sum2);
                    vst1q_s32(outptr0 + out_hstep * 3, _sum3);
                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
            }

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            int32x4_t _sum0;
            int32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                // aabbccdd eeffgghh

                // 00112233 -> 00110011 22332233
                // 11001100 33223322

                int32x2x2_t _pBB = vzip_s32(vreinterpret_s32_s8(_pB), vreinterpret_s32_s8(_pB));
                int8x16_t _pB02 = vreinterpretq_s8_s32(vcombine_s32(_pBB.val[0], _pBB.val[1]));

                int8x16_t _pB13 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB02)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), vget_low_s8(_pB02));
                int16x8_t _s1 = vmull_s8(vget_low_s8(_pA), vget_low_s8(_pB13));
                _s0 = vmlal_s8(_s0, vget_high_s8(_pA), vget_high_s8(_pB02));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA), vget_high_s8(_pB13));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);

                pA += 16;
                pB += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));

                // aabbccdd

                // 00110011
                // 11001100
                int8x8_t _pB1 = vext_s8(_pB0, _pB0, 2);

                int16x8_t _s0 = vmull_s8(_pA, _pB0);
                int16x8_t _s1 = vmull_s8(_pA, _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);

                pA += 8;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                int8x8_t _pB0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));

                // abcd abcd

                // 0101 0101 -> 0101 1010

                int8x8_t _pB1 = vext_s8(_pB0, _pB0, 1);
                int8x8_t _pB = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_pB0), vreinterpret_s32_s8(_pB1)).val[0]);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                // from
                //      a0 b1 c0 d1
                //      a1 b0 c1 d0
                if (out_elempack == 4)
                {
                    // to
                    //      a0 b0 c0 d0
                    //      a1 b1 c1 d1
                    {
                        _sum1 = vrev64q_s32(_sum1);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 c0 c1
                    //      b0 b1 d0 d1
                    {
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                    }

                    vst1_s32(outptr0, vget_low_s32(_sum0));
                    vst1_s32(outptr0 + out_hstep, vget_low_s32(_sum1));
                    vst1_s32(outptr0 + out_hstep * 2, vget_high_s32(_sum0));
                    vst1_s32(outptr0 + out_hstep * 3, vget_high_s32(_sum1));
                    outptr0 += 2;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            int32x4_t _sum0;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));
                int8x8_t _pB1 = vreinterpret_s8_s16(vld1_dup_s16((const short*)(pB + 2)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), _pB0);
                _s0 = vmlal_s8(_s0, vget_high_s8(_pA), _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);

                pA += 16;
                pB += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vpadalq_s16(_sum0, _s0);

                pA += 8;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                int8x8_t _pB = vld1_dup_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_s32(outptr0, _sum0);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    outptr0[0] = vgetq_lane_s32(_sum0, 0);
                    outptr0[out_hstep] = vgetq_lane_s32(_sum0, 1);
                    outptr0[out_hstep * 2] = vgetq_lane_s32(_sum0, 2);
                    outptr0[out_hstep * 3] = vgetq_lane_s32(_sum0, 3);
                    outptr0++;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum0);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        int* outptr0 = (int*)top_blob + (i + ii) * out_hstep + j;

        const signed char* pB = pBT;

        int jj = 0;
#if __riscv_vector
        for (; jj + 7 < max_jj; jj += 8)
        {
            int32x4_t _sum0;
            int32x4_t _sum1;
            int32x4_t _sum2;
            int32x4_t _sum3;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
                _sum2 = vld1q_s32(outptr + 8);
                _sum3 = vld1q_s32(outptr + 12);
            }

            const signed char* pA = pAT;
            int kk = 0;
            {
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);

                    int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 0));
                    int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 1));
                    int8x8_t _pA2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 2));
                    int8x8_t _pA3 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 3));

                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                    int16x8_t _s1 = vmull_s8(_pA0, vget_high_s8(_pB0));
                    int16x8_t _s2 = vmull_s8(_pA1, vget_low_s8(_pB0));
                    int16x8_t _s3 = vmull_s8(_pA1, vget_high_s8(_pB0));
                    _s0 = vmlal_s8(_s0, _pA2, vget_low_s8(_pB1));
                    _s1 = vmlal_s8(_s1, _pA2, vget_high_s8(_pB1));
                    _s2 = vmlal_s8(_s2, _pA3, vget_low_s8(_pB1));
                    _s3 = vmlal_s8(_s3, _pA3, vget_high_s8(_pB1));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);

                    pA += 8;
                    pB += 32;
                }
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int16x4_t _pA = vreinterpret_s16_s32(vld1_dup_s32((const int*)pA));
                int8x16_t _pB = vld1q_s8(pB);

                int16x4x2_t _pA01 = vuzp_s16(_pA, _pA);
                int8x8_t _pA0 = vreinterpret_s8_s16(_pA01.val[0]);
                int8x8_t _pA1 = vreinterpret_s8_s16(_pA01.val[1]);

                int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB));
                int16x8_t _s1 = vmull_s8(_pA0, vget_high_s8(_pB));
                int16x8_t _s2 = vmull_s8(_pA1, vget_low_s8(_pB));
                int16x8_t _s3 = vmull_s8(_pA1, vget_high_s8(_pB));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);

                pA += 4;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vreinterpret_s8_s16(vld1_dup_s16((const short*)pA));
                int8x8_t _pB = vld1_s8(pB);

                int8x8x2_t _pA01 = vuzp_s8(_pA, _pA);

                int16x8_t _s0 = vmull_s8(_pA01.val[0], _pB);
                int16x8_t _s1 = vmull_s8(_pA01.val[1], _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));

                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + out_hstep, _sum2);
                    vst1q_s32(outptr0 + out_hstep + 4, _sum3);
                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
            }

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            int32x4_t _sum0;
            int32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
            }

            const signed char* pA = pAT;
            int kk = 0;
            {
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB = vld1q_s8(pB);

                    int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 0));
                    int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 1));
                    int8x8_t _pA2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 2));
                    int8x8_t _pA3 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 3));

                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB));
                    int16x8_t _s1 = vmull_s8(_pA1, vget_low_s8(_pB));
                    _s0 = vmlal_s8(_s0, _pA2, vget_high_s8(_pB));
                    _s1 = vmlal_s8(_s1, _pA3, vget_high_s8(_pB));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);

                    pA += 8;
                    pB += 16;
                }
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int16x4_t _pA = vreinterpret_s16_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pA)), 0));
                int8x8_t _pB = vld1_s8(pB);

                int16x4x2_t _pA01 = vuzp_s16(_pA, _pA);
                int8x8_t _pA0 = vreinterpret_s8_s16(_pA01.val[0]);
                int8x8_t _pA1 = vreinterpret_s8_s16(_pA01.val[1]);

                int16x8_t _s0 = vmull_s8(_pA0, _pB);
                int16x8_t _s1 = vmull_s8(_pA1, _pB);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);

                pA += 4;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vreinterpret_s8_s16(vld1_dup_s16((const short*)pA));
                int8x8_t _pB = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pB)), 0));

                _pA = vzip_s8(_pA, _pA).val[0];
                _pA = vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_pA), vreinterpret_s16_s8(_pA)).val[0]);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + out_hstep, _sum1);
                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
#endif // __riscv_vector
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __riscv_vector
            int32x4_t _sum;

            if (k == 0)
            {
                _sum = vdupq_n_s32(0);
            }
            else
            {
                _sum = vld1q_s32(outptr);
            }

            const signed char* pA = pAT;
            int kk = 0;

            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                int16x4x2_t _pA01 = vzip_s16(vreinterpret_s16_s8(_pA), vreinterpret_s16_s8(_pA));
                int32x2x2_t _pB01 = vzip_s32(vreinterpret_s32_s8(_pB), vreinterpret_s32_s8(_pB));

                int16x8_t _s0 = vmull_s8(vreinterpret_s8_s16(_pA01.val[0]), vreinterpret_s8_s32(_pB01.val[0]));
                _s0 = vmlal_s8(_s0, vreinterpret_s8_s16(_pA01.val[1]), vreinterpret_s8_s32(_pB01.val[1]));
                _sum = vpadalq_s16(_sum, _s0);

                pA += 8;
                pB += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                _pA = vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_pA), vreinterpret_s16_s8(_pA)).val[0]);
                _pB = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_pB), vreinterpret_s32_s8(_pB)).val[0]);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum = vpadalq_s16(_sum, _s0);

                pA += 4;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vreinterpret_s8_s16(vld1_dup_s16((const short*)pA));
                int8x8_t _pB = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vld1_s8(pB)), 0));

                _pA = vzip_s8(_pA, _pA).val[0];

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum = vaddw_s16(_sum, vget_low_s16(_s0));

                pA += 2;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_s32(outptr0, vget_low_s32(_sum));
                    vst1_s32(outptr0 + out_hstep, vget_high_s32(_sum));
                    outptr0 += 2;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum);
            }

            outptr += 4;
#else  // __riscv_vector
            int sum00;
            int sum10;
            int sum01;
            int sum11;

            if (k == 0)
            {
                sum00 = 0;
                sum10 = 0;
                sum01 = 0;
                sum11 = 0;
            }
            else
            {
                sum00 = outptr[0];
                sum10 = outptr[1];
                sum01 = outptr[2];
                sum11 = outptr[3];
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum00 += pA[0] * pB[0];
                sum10 += pA[1] * pB[0];
                sum01 += pA[0] * pB[1];
                sum11 += pA[1] * pB[1];

                pA += 2;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum00;
                    outptr0[1] = sum01;
                    outptr0[out_hstep] = sum10;
                    outptr0[out_hstep + 1] = sum11;
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum10;
                outptr[2] = sum01;
                outptr[3] = sum11;
            }

            outptr += 4;
#endif // __riscv_vector
        }
        for (; jj < max_jj; jj += 1)
        {
#if __riscv_vector
            int32x2_t _sum;

            if (k == 0)
            {
                _sum = vdup_n_s32(0);
            }
            else
            {
                _sum = vld1_s32(outptr);
            }
#else  // __riscv_vector
            int sum0;
            int sum1;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }
#endif // __riscv_vector

            const signed char* pA = pAT;
            int kk = 0;
#if __riscv_vector
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x8_t _pB = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pB)), 0));

                    _pB = vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_pB), vreinterpret_s16_s8(_pB)).val[0]);

                    int16x8_t _s0 = vmull_s8(_pA, _pB);
                    _sum0 = vpadalq_s16(_sum0, _s0);

                    pA += 8;
                    pB += 4;
                }
                int32x2_t _ss = vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                _sum = vadd_s32(_sum, _ss);
            }
            int sum0 = vget_lane_s32(_sum, 0);
            int sum1 = vget_lane_s32(_sum, 1);
            for (; kk + 1 < max_kk; kk += 2)
            {
                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum1 += pA[2] * pB[0];
                sum1 += pA[3] * pB[1];
                pA += 4;
                pB += 2;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[0];
                pA += 2;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[out_hstep] = sum1;
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii += 1)
    {
        int* outptr0 = (int*)top_blob + (i + ii) * out_hstep + j;

        const signed char* pB = pBT;

        int jj = 0;
#if __riscv_vector
        for (; jj + 7 < max_jj; jj += 8)
        {
            int32x4_t _sum0;
            int32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x8_t _pA = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pA)), 0));
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB1 = vld1q_s8(pB + 16);

                int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 0));
                int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 1));
                int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                int16x8_t _s1 = vmull_s8(_pA0, vget_high_s8(_pB0));
                _s0 = vmlal_s8(_s0, _pA1, vget_low_s8(_pB1));
                _s1 = vmlal_s8(_s1, _pA1, vget_high_s8(_pB1));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);

                pA += 4;
                pB += 32;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8_t _pA = vreinterpret_s8_s16(vld1_dup_s16((const short*)pA));
                int8x16_t _pB = vld1q_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, vget_low_s8(_pB));
                int16x8_t _s1 = vmull_s8(_pA, vget_high_s8(_pB));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);

                pA += 2;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vld1_dup_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            int32x4_t _sum0;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x16_t _pB = vld1q_s8(pB);

                int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 0));
                int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 1));
                int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB));
                _s0 = vmlal_s8(_s0, _pA1, vget_high_s8(_pB));
                _sum0 = vpadalq_s16(_sum0, _s0);

                pA += 4;
                pB += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8_t _pA = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vld1_s8(pA)), 0));
                int8x8_t _pB = vld1_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vpadalq_s16(_sum0, _s0);

                pA += 2;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vld1_dup_s8(pA);
                int8x8_t _pB = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pB)), 0));

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_s32(outptr0, _sum0);
                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum0);
            }

            outptr += 4;
        }
#endif // __riscv_vector
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __riscv_vector
            int32x2_t _sum;

            if (k == 0)
            {
                _sum = vdup_n_s32(0);
            }
            else
            {
                _sum = vld1_s32(outptr);
            }
#else  // __riscv_vector
            int sum0;
            int sum1;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }
#endif // __riscv_vector

            const signed char* pA = pAT;
            int kk = 0;
#if __riscv_vector
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int8x8_t _pA = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pA)), 0));
                    int8x8_t _pB = vld1_s8(pB);

                    _pA = vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_pA), vreinterpret_s16_s8(_pA)).val[0]);

                    int16x8_t _s0 = vmull_s8(_pA, _pB);
                    _sum0 = vpadalq_s16(_sum0, _s0);

                    pA += 4;
                    pB += 8;
                }
                int32x2_t _ss = vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                _sum = vadd_s32(_sum, _ss);
            }
            int sum0 = vget_lane_s32(_sum, 0);
            int sum1 = vget_lane_s32(_sum, 1);
            for (; kk + 1 < max_kk; kk += 2)
            {
                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum1 += pA[0] * pB[2];
                sum1 += pA[1] * pB[3];
                pA += 2;
                pB += 4;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];
                pA += 1;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[1] = sum1;
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            int sum;

            if (k == 0)
            {
                sum = 0;
            }
            else
            {
                sum = outptr[0];
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __riscv_vector
            int32x4_t _sum = vdupq_n_s32(0);
            for (; kk + 15 < max_kk; kk += 16)
            {
                int8x16_t _pA = vld1q_s8(pA);
                int8x16_t _pB = vld1q_s8(pB);

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), vget_low_s8(_pB));
                _s0 = vmlal_s8(_s0, vget_high_s8(_pA), vget_high_s8(_pB));
                _sum = vpadalq_s16(_sum, _s0);

                pA += 16;
                pB += 16;
            }
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum = vpadalq_s16(_sum, _s0);

                pA += 8;
                pB += 8;
            }
            sum += vaddvq_s32(_sum);
#endif // __riscv_vector
            for (; kk < max_kk; kk += 1)
            {
                sum += pA[0] * pB[0];
                pA += 1;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum;
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum;
            }

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void convolution_im2col_gemm_get_optimal_tile_mnk_int8(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size_int8 = (int)(get_cpu_level2_cache_size() / sizeof(signed char));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve K
    {
        // try not to split K
#if __riscv_vector
        int tile_size = (l2_cache_size_int8 - 16) / 8;
#else
        int tile_size = (l2_cache_size_int8 - 2) / 3;
#endif

#if __riscv_vector
        TILE_K = std::max(4, tile_size / 4 * 4);
#else
        TILE_K = std::max(2, tile_size / 2 * 2);
#endif

        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __riscv_vector
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif
    }

    // solve M
    {
#if __riscv_vector
        int nn_M = (M + 31) / 32;
#else
        int nn_M = (M + 7) / 8;
#endif

#if __riscv_vector
        TILE_M = std::max(8, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::max(2, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    {
        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __riscv_vector
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif

        if (nT > 1)
        {
#if __riscv_vector
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#else
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
        }
    }

    if (N > 0)
    {
        int tile_size;
        if (TILE_K >= K)
        {
            tile_size = (l2_cache_size_int8 - TILE_M * TILE_K) / TILE_K;
        }
        else
        {
            tile_size = (l2_cache_size_int8 - TILE_M * TILE_K) / (TILE_M * 4 + TILE_K);
        }

#if __riscv_vector
        TILE_N = std::max(4, tile_size / 4 * 4);
#else
        TILE_N = std::max(1, tile_size);
#endif

        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __riscv_vector
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif
    }
}

static void convolution_im2col_input_tile_conv1x1s1d1_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int elempack = bottom_blob.elempack;

    signed char* pp = B;

    int jj = 0;
#if __riscv_vector
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;
            const size_t cstep = bottom_blob.cstep * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                int16x8x4_t _r0 = vld4q_s16((const short*)p0);
                vst1q_s16((short*)pp, _r0.val[0]);
                vst1q_s16((short*)(pp + 16), _r0.val[1]);
                vst1q_s16((short*)(pp + 32), _r0.val[2]);
                vst1q_s16((short*)(pp + 48), _r0.val[3]);
                pp += 64;
                p0 += cstep;
            }
        }

        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);
            const size_t cstep = bottom_blob.cstep;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8x2_t _r01;
                _r01.val[0] = vld1_s8(p0);
                _r01.val[1] = vld1_s8(p0 + cstep);
                vst2_s8(pp, _r01);
                pp += 16;
                p0 += cstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                vst1_s8(pp, vld1_s8(p0));
                pp += 8;
                p0 += cstep;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;
            const size_t cstep = bottom_blob.cstep * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                int16x4x4_t _r0123;
                _r0123.val[0] = vreinterpret_s16_s8(vld1_s8(p0));
                _r0123.val[1] = vreinterpret_s16_s8(vld1_s8(p0 + 8));
                _r0123.val[2] = vreinterpret_s16_s8(vld1_s8(p0 + 16));
                _r0123.val[3] = vreinterpret_s16_s8(vld1_s8(p0 + 24));
                vst4_s16((short*)pp, _r0123);
                pp += 32;
                p0 += cstep;
            }
        }

        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);
            const size_t cstep = bottom_blob.cstep;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[cstep + 0];
                pp[2] = p0[1];
                pp[3] = p0[cstep + 1];
                pp[4] = p0[2];
                pp[5] = p0[cstep + 2];
                pp[6] = p0[3];
                pp[7] = p0[cstep + 3];
                pp += 8;
                p0 += cstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += cstep;
            }
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __riscv_vector
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;
            const size_t cstep = bottom_blob.cstep * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                int16x4x2_t _r01;
                _r01.val[0] = vreinterpret_s16_s8(vld1_s8(p0));
                _r01.val[1] = vreinterpret_s16_s8(vld1_s8(p0 + 8));
                vst2_s16((short*)pp, _r01);
                pp += 16;
                p0 += cstep;
            }
        }
#endif // __riscv_vector

        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);
            const size_t cstep = bottom_blob.cstep;

            int kk = 0;
#if __riscv_vector
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[cstep];
                pp[2] = p0[1];
                pp[3] = p0[cstep + 1];
                pp += 4;
                p0 += cstep * 2;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += cstep;
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
#if __riscv_vector
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;
            const size_t cstep = bottom_blob.cstep * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                vst1_s8(pp, vld1_s8(p0));
                pp += 8;
                p0 += cstep;
            }
        }
#endif // __riscv_vector

        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);
            const size_t cstep = bottom_blob.cstep;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += cstep;
            }
        }
    }
}

template<int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h>
void convolution_im2col_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int w = bottom_blob.w;
    // const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int outw = (w - kernel_extent_w) / stride_w + 1;

    // j max_jj     outw*outh    split w and h

    // k max_kk     pa*maxk*(inch/pa)    split inch

    // k/max_kk shall be multiple of maxk

    const int maxk = kernel_w * kernel_h;

    signed char* pp = B;

    int jj = 0;
#if __riscv_vector
    for (; jj + 7 < max_jj; jj += 8)
    {
        int dy0 = (j + jj) / outw * stride_h;
        int dy1 = (j + jj + 1) / outw * stride_h;
        int dy2 = (j + jj + 2) / outw * stride_h;
        int dy3 = (j + jj + 3) / outw * stride_h;
        int dy4 = (j + jj + 4) / outw * stride_h;
        int dy5 = (j + jj + 5) / outw * stride_h;
        int dy6 = (j + jj + 6) / outw * stride_h;
        int dy7 = (j + jj + 7) / outw * stride_h;
        int dx0 = (j + jj) % outw * stride_w;
        int dx1 = (j + jj + 1) % outw * stride_w;
        int dx2 = (j + jj + 2) % outw * stride_w;
        int dx3 = (j + jj + 3) % outw * stride_w;
        int dx4 = (j + jj + 4) % outw * stride_w;
        int dx5 = (j + jj + 5) % outw * stride_w;
        int dx6 = (j + jj + 6) % outw * stride_w;
        int dx7 = (j + jj + 7) % outw * stride_w;

        if (dy0 == dy7)
        {
            int kk = 0;
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    if (stride_w == 1)
                    {
                        int8x8x2_t _r01;
                        _r01.val[0] = vld1_s8(sptr0);
                        _r01.val[1] = vld1_s8(sptr1);
                        vst2_s8(pp, _r01);
                        pp += 16;
                    }
                    else if (stride_w == 2)
                    {
                        int8x16_t _r0 = vld1q_s8(sptr0);
                        int8x16_t _r1 = vld1q_s8(sptr1);
                        int8x16_t _r01 = vtrnq_s8(_r0, _r1).val[0];
                        vst1q_s8(pp, _r01);
                        pp += 16;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr0[stride_w];
                        pp[3] = sptr1[stride_w];
                        pp[4] = sptr0[stride_w * 2];
                        pp[5] = sptr1[stride_w * 2];
                        pp[6] = sptr0[stride_w * 3];
                        pp[7] = sptr1[stride_w * 3];
                        pp[8] = sptr0[stride_w * 4];
                        pp[9] = sptr1[stride_w * 4];
                        pp[10] = sptr0[stride_w * 5];
                        pp[11] = sptr1[stride_w * 5];
                        pp[12] = sptr0[stride_w * 6];
                        pp[13] = sptr1[stride_w * 6];
                        pp[14] = sptr0[stride_w * 7];
                        pp[15] = sptr1[stride_w * 7];
                        pp += 16;
                    }
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;

                const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

                if (elempack == 8)
                {
                    int16x4_t _r0 = vreinterpret_s16_s8(vld1_s8(sptr));
                    int16x4_t _r1 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 8));
                    int16x4_t _r2 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 16));
                    int16x4_t _r3 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 24));
                    int16x4_t _r4 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 32));
                    int16x4_t _r5 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 40));
                    int16x4_t _r6 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 48));
                    int16x4_t _r7 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 56));
                    int16x4x2_t _r01 = vzip_s16(_r0, _r1);
                    int16x4x2_t _r23 = vzip_s16(_r2, _r3);
                    int16x4x2_t _r45 = vzip_s16(_r4, _r5);
                    int16x4x2_t _r67 = vzip_s16(_r6, _r7);
                    int32x4x4_t _r0123;
                    _r0123.val[0] = vreinterpretq_s32_s16(vcombine_s16(_r01.val[0], _r01.val[1]));
                    _r0123.val[1] = vreinterpretq_s32_s16(vcombine_s16(_r23.val[0], _r23.val[1]));
                    _r0123.val[2] = vreinterpretq_s32_s16(vcombine_s16(_r45.val[0], _r45.val[1]));
                    _r0123.val[3] = vreinterpretq_s32_s16(vcombine_s16(_r67.val[0], _r67.val[1]));
                    vst4q_s32((int*)pp, _r0123);
                    pp += 64;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp[2] = sptr[stride_w * 2];
                    pp[3] = sptr[stride_w * 3];
                    pp[4] = sptr[stride_w * 4];
                    pp[5] = sptr[stride_w * 5];
                    pp[6] = sptr[stride_w * 6];
                    pp[7] = sptr[stride_w * 7];
                    pp += 8;
                }
            }
        }
        else
        {
            int kk = 0;
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int x02 = dx2 + dilation_w * v0;
                    int x03 = dx3 + dilation_w * v0;
                    int x04 = dx4 + dilation_w * v0;
                    int x05 = dx5 + dilation_w * v0;
                    int x06 = dx6 + dilation_w * v0;
                    int x07 = dx7 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int y02 = dy2 + dilation_h * u0;
                    int y03 = dy3 + dilation_h * u0;
                    int y04 = dy4 + dilation_h * u0;
                    int y05 = dy5 + dilation_h * u0;
                    int y06 = dy6 + dilation_h * u0;
                    int y07 = dy7 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int x12 = dx2 + dilation_w * v1;
                    int x13 = dx3 + dilation_w * v1;
                    int x14 = dx4 + dilation_w * v1;
                    int x15 = dx5 + dilation_w * v1;
                    int x16 = dx6 + dilation_w * v1;
                    int x17 = dx7 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int y12 = dy2 + dilation_h * u1;
                    int y13 = dy3 + dilation_h * u1;
                    int y14 = dy4 + dilation_h * u1;
                    int y15 = dy5 + dilation_h * u1;
                    int y16 = dy6 + dilation_h * u1;
                    int y17 = dy7 + dilation_h * u1;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;
                    const signed char* sptr04 = img0.row<const signed char>(y04) + x04;
                    const signed char* sptr05 = img0.row<const signed char>(y05) + x05;
                    const signed char* sptr06 = img0.row<const signed char>(y06) + x06;
                    const signed char* sptr07 = img0.row<const signed char>(y07) + x07;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;
                    const signed char* sptr14 = img1.row<const signed char>(y14) + x14;
                    const signed char* sptr15 = img1.row<const signed char>(y15) + x15;
                    const signed char* sptr16 = img1.row<const signed char>(y16) + x16;
                    const signed char* sptr17 = img1.row<const signed char>(y17) + x17;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr01[0];
                    pp[3] = sptr11[0];
                    pp[4] = sptr02[0];
                    pp[5] = sptr12[0];
                    pp[6] = sptr03[0];
                    pp[7] = sptr13[0];
                    pp[8] = sptr04[0];
                    pp[9] = sptr14[0];
                    pp[10] = sptr05[0];
                    pp[11] = sptr15[0];
                    pp[12] = sptr06[0];
                    pp[13] = sptr16[0];
                    pp[14] = sptr07[0];
                    pp[15] = sptr17[0];
                    pp += 16;
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int x1 = dx1 + dilation_w * v;
                int x2 = dx2 + dilation_w * v;
                int x3 = dx3 + dilation_w * v;
                int x4 = dx4 + dilation_w * v;
                int x5 = dx5 + dilation_w * v;
                int x6 = dx6 + dilation_w * v;
                int x7 = dx7 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;
                int y1 = dy1 + dilation_h * u;
                int y2 = dy2 + dilation_h * u;
                int y3 = dy3 + dilation_h * u;
                int y4 = dy4 + dilation_h * u;
                int y5 = dy5 + dilation_h * u;
                int y6 = dy6 + dilation_h * u;
                int y7 = dy7 + dilation_h * u;

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;
                const signed char* sptr2 = img.row<const signed char>(y2) + x2 * elempack;
                const signed char* sptr3 = img.row<const signed char>(y3) + x3 * elempack;
                const signed char* sptr4 = img.row<const signed char>(y4) + x4 * elempack;
                const signed char* sptr5 = img.row<const signed char>(y5) + x5 * elempack;
                const signed char* sptr6 = img.row<const signed char>(y6) + x6 * elempack;
                const signed char* sptr7 = img.row<const signed char>(y7) + x7 * elempack;

                if (elempack == 8)
                {
                    int16x4_t _r0 = vreinterpret_s16_s8(vld1_s8(sptr0));
                    int16x4_t _r1 = vreinterpret_s16_s8(vld1_s8(sptr1));
                    int16x4_t _r2 = vreinterpret_s16_s8(vld1_s8(sptr2));
                    int16x4_t _r3 = vreinterpret_s16_s8(vld1_s8(sptr3));
                    int16x4_t _r4 = vreinterpret_s16_s8(vld1_s8(sptr4));
                    int16x4_t _r5 = vreinterpret_s16_s8(vld1_s8(sptr5));
                    int16x4_t _r6 = vreinterpret_s16_s8(vld1_s8(sptr6));
                    int16x4_t _r7 = vreinterpret_s16_s8(vld1_s8(sptr7));
                    int16x4x2_t _r01 = vzip_s16(_r0, _r1);
                    int16x4x2_t _r23 = vzip_s16(_r2, _r3);
                    int16x4x2_t _r45 = vzip_s16(_r4, _r5);
                    int16x4x2_t _r67 = vzip_s16(_r6, _r7);
                    int32x4x4_t _r0123;
                    _r0123.val[0] = vreinterpretq_s32_s16(vcombine_s16(_r01.val[0], _r01.val[1]));
                    _r0123.val[1] = vreinterpretq_s32_s16(vcombine_s16(_r23.val[0], _r23.val[1]));
                    _r0123.val[2] = vreinterpretq_s32_s16(vcombine_s16(_r45.val[0], _r45.val[1]));
                    _r0123.val[3] = vreinterpretq_s32_s16(vcombine_s16(_r67.val[0], _r67.val[1]));
                    vst4q_s32((int*)pp, _r0123);
                    pp += 64;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp[4] = sptr4[0];
                    pp[5] = sptr5[0];
                    pp[6] = sptr6[0];
                    pp[7] = sptr7[0];
                    pp += 8;
                }
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        int dy0 = (j + jj) / outw * stride_h;
        int dy1 = (j + jj + 1) / outw * stride_h;
        int dy2 = (j + jj + 2) / outw * stride_h;
        int dy3 = (j + jj + 3) / outw * stride_h;
        int dx0 = (j + jj) % outw * stride_w;
        int dx1 = (j + jj + 1) % outw * stride_w;
        int dx2 = (j + jj + 2) % outw * stride_w;
        int dx3 = (j + jj + 3) % outw * stride_w;

        if (dy0 == dy3)
        {
            int kk = 0;
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    if (stride_w == 1)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r01 = vzip_s8(_r0, _r1).val[0];
                        vst1_s8(pp, _r01);
                        pp += 8;
                    }
                    else if (stride_w == 2)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r01 = vtrn_s8(_r0, _r1).val[0];
                        vst1_s8(pp, _r01);
                        pp += 8;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr0[stride_w];
                        pp[3] = sptr1[stride_w];
                        pp[4] = sptr0[stride_w * 2];
                        pp[5] = sptr1[stride_w * 2];
                        pp[6] = sptr0[stride_w * 3];
                        pp[7] = sptr1[stride_w * 3];
                        pp += 8;
                    }
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;

                const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

                if (elempack == 8)
                {
                    int16x4x4_t _r0123;
                    _r0123.val[0] = vreinterpret_s16_s8(vld1_s8(sptr));
                    _r0123.val[1] = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 8));
                    _r0123.val[2] = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 16));
                    _r0123.val[3] = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 24));
                    vst4_s16((short*)pp, _r0123);
                    pp += 32;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp[2] = sptr[stride_w * 2];
                    pp[3] = sptr[stride_w * 3];
                    pp += 4;
                }
            }
        }
        else
        {
            int kk = 0;
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int x02 = dx2 + dilation_w * v0;
                    int x03 = dx3 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int y02 = dy2 + dilation_h * u0;
                    int y03 = dy3 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int x12 = dx2 + dilation_w * v1;
                    int x13 = dx3 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int y12 = dy2 + dilation_h * u1;
                    int y13 = dy3 + dilation_h * u1;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr01[0];
                    pp[3] = sptr11[0];
                    pp[4] = sptr02[0];
                    pp[5] = sptr12[0];
                    pp[6] = sptr03[0];
                    pp[7] = sptr13[0];
                    pp += 8;
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int x1 = dx1 + dilation_w * v;
                int x2 = dx2 + dilation_w * v;
                int x3 = dx3 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;
                int y1 = dy1 + dilation_h * u;
                int y2 = dy2 + dilation_h * u;
                int y3 = dy3 + dilation_h * u;

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;
                const signed char* sptr2 = img.row<const signed char>(y2) + x2 * elempack;
                const signed char* sptr3 = img.row<const signed char>(y3) + x3 * elempack;

                if (elempack == 8)
                {
                    int16x4x4_t _r0123;
                    _r0123.val[0] = vreinterpret_s16_s8(vld1_s8(sptr0));
                    _r0123.val[1] = vreinterpret_s16_s8(vld1_s8(sptr1));
                    _r0123.val[2] = vreinterpret_s16_s8(vld1_s8(sptr2));
                    _r0123.val[3] = vreinterpret_s16_s8(vld1_s8(sptr3));
                    vst4_s16((short*)pp, _r0123);
                    pp += 32;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp += 4;
                }
            }
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
        int dy0 = (j + jj) / outw * stride_h;
        int dy1 = (j + jj + 1) / outw * stride_h;
        int dx0 = (j + jj) % outw * stride_w;
        int dx1 = (j + jj + 1) % outw * stride_w;

        if (dy0 == dy1)
        {
            int kk = 0;
#if __riscv_vector
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr0[stride_w];
                    pp[3] = sptr1[stride_w];
                    pp += 4;
                }
            }
#endif // __riscv_vector
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;

                const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

#if __riscv_vector
                if (elempack == 8)
                {
                    int16x4x2_t _r01;
                    _r01.val[0] = vreinterpret_s16_s8(vld1_s8(sptr));
                    _r01.val[1] = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 8));
                    vst2_s16((short*)pp, _r01);
                    pp += 16;
                }
#endif // __riscv_vector
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp += 2;
                }
            }
        }
        else
        {
            int kk = 0;
#if __riscv_vector
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr01[0];
                    pp[3] = sptr11[0];
                    pp += 4;
                }
            }
#endif // __riscv_vector
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int x1 = dx1 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;
                int y1 = dy1 + dilation_h * u;

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;

#if __riscv_vector
                if (elempack == 8)
                {
                    int16x4x2_t _r01;
                    _r01.val[0] = vreinterpret_s16_s8(vld1_s8(sptr0));
                    _r01.val[1] = vreinterpret_s16_s8(vld1_s8(sptr1));
                    vst2_s16((short*)pp, _r01);
                    pp += 16;
                }
#endif // __riscv_vector
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp += 2;
                }
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
        int dy = (j + jj) / outw * stride_h;
        int dx = (j + jj) % outw * stride_w;

        int kk = 0;
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x = dx + dilation_w * v;
            int y = dy + dilation_h * u;

            const signed char* sptr = img.row<const signed char>(y) + x * elempack;

#if __riscv_vector
            if (elempack == 8)
            {
                vst1_s8(pp, vld1_s8(sptr));
                pp += 8;
            }
#endif // __riscv_vector
            if (elempack == 1)
            {
                pp[0] = sptr[0];
                pp += 1;
            }
        }
    }
}

template void convolution_im2col_input_tile_int8<1, 1, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<3, 3, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<3, 3, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<5, 5, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<5, 5, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<7, 7, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);

static void convolution_im2col_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1_int8(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 1 && kernel_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_int8<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_int8<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_int8<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_int8<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_int8<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_int8<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    const int w = bottom_blob.w;
    // const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int outw = (w - kernel_extent_w) / stride_w + 1;

    // j max_jj     outw*outh    split w and h

    // k max_kk     pa*maxk*(inch/pa)    split inch

    // k/max_kk shall be multiple of maxk

    const int maxk = kernel_w * kernel_h;

    signed char* pp = B;

    int jj = 0;
#if __riscv_vector
    for (; jj + 7 < max_jj; jj += 8)
    {
        int dy0 = (j + jj) / outw * stride_h;
        int dy1 = (j + jj + 1) / outw * stride_h;
        int dy2 = (j + jj + 2) / outw * stride_h;
        int dy3 = (j + jj + 3) / outw * stride_h;
        int dy4 = (j + jj + 4) / outw * stride_h;
        int dy5 = (j + jj + 5) / outw * stride_h;
        int dy6 = (j + jj + 6) / outw * stride_h;
        int dy7 = (j + jj + 7) / outw * stride_h;
        int dx0 = (j + jj) % outw * stride_w;
        int dx1 = (j + jj + 1) % outw * stride_w;
        int dx2 = (j + jj + 2) % outw * stride_w;
        int dx3 = (j + jj + 3) % outw * stride_w;
        int dx4 = (j + jj + 4) % outw * stride_w;
        int dx5 = (j + jj + 5) % outw * stride_w;
        int dx6 = (j + jj + 6) % outw * stride_w;
        int dx7 = (j + jj + 7) % outw * stride_w;

        if (dy0 == dy7)
        {
            int kk = 0;
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    if (stride_w == 1)
                    {
                        int8x8x2_t _r01;
                        _r01.val[0] = vld1_s8(sptr0);
                        _r01.val[1] = vld1_s8(sptr1);
                        vst2_s8(pp, _r01);
                        pp += 16;
                    }
                    else if (stride_w == 2)
                    {
                        int8x16_t _r0 = vld1q_s8(sptr0);
                        int8x16_t _r1 = vld1q_s8(sptr1);
                        int8x16_t _r01 = vtrnq_s8(_r0, _r1).val[0];
                        vst1q_s8(pp, _r01);
                        pp += 16;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr0[stride_w];
                        pp[3] = sptr1[stride_w];
                        pp[4] = sptr0[stride_w * 2];
                        pp[5] = sptr1[stride_w * 2];
                        pp[6] = sptr0[stride_w * 3];
                        pp[7] = sptr1[stride_w * 3];
                        pp[8] = sptr0[stride_w * 4];
                        pp[9] = sptr1[stride_w * 4];
                        pp[10] = sptr0[stride_w * 5];
                        pp[11] = sptr1[stride_w * 5];
                        pp[12] = sptr0[stride_w * 6];
                        pp[13] = sptr1[stride_w * 6];
                        pp[14] = sptr0[stride_w * 7];
                        pp[15] = sptr1[stride_w * 7];
                        pp += 16;
                    }
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;

                const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

                if (elempack == 8)
                {
                    int16x4_t _r0 = vreinterpret_s16_s8(vld1_s8(sptr));
                    int16x4_t _r1 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 8));
                    int16x4_t _r2 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 16));
                    int16x4_t _r3 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 24));
                    int16x4_t _r4 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 32));
                    int16x4_t _r5 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 40));
                    int16x4_t _r6 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 48));
                    int16x4_t _r7 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 56));
                    int16x4x2_t _r01 = vzip_s16(_r0, _r1);
                    int16x4x2_t _r23 = vzip_s16(_r2, _r3);
                    int16x4x2_t _r45 = vzip_s16(_r4, _r5);
                    int16x4x2_t _r67 = vzip_s16(_r6, _r7);
                    int32x4x4_t _r0123;
                    _r0123.val[0] = vreinterpretq_s32_s16(vcombine_s16(_r01.val[0], _r01.val[1]));
                    _r0123.val[1] = vreinterpretq_s32_s16(vcombine_s16(_r23.val[0], _r23.val[1]));
                    _r0123.val[2] = vreinterpretq_s32_s16(vcombine_s16(_r45.val[0], _r45.val[1]));
                    _r0123.val[3] = vreinterpretq_s32_s16(vcombine_s16(_r67.val[0], _r67.val[1]));
                    vst4q_s32((int*)pp, _r0123);
                    pp += 64;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp[2] = sptr[stride_w * 2];
                    pp[3] = sptr[stride_w * 3];
                    pp[4] = sptr[stride_w * 4];
                    pp[5] = sptr[stride_w * 5];
                    pp[6] = sptr[stride_w * 6];
                    pp[7] = sptr[stride_w * 7];
                    pp += 8;
                }
            }
        }
        else
        {
            int kk = 0;
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int x02 = dx2 + dilation_w * v0;
                    int x03 = dx3 + dilation_w * v0;
                    int x04 = dx4 + dilation_w * v0;
                    int x05 = dx5 + dilation_w * v0;
                    int x06 = dx6 + dilation_w * v0;
                    int x07 = dx7 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int y02 = dy2 + dilation_h * u0;
                    int y03 = dy3 + dilation_h * u0;
                    int y04 = dy4 + dilation_h * u0;
                    int y05 = dy5 + dilation_h * u0;
                    int y06 = dy6 + dilation_h * u0;
                    int y07 = dy7 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int x12 = dx2 + dilation_w * v1;
                    int x13 = dx3 + dilation_w * v1;
                    int x14 = dx4 + dilation_w * v1;
                    int x15 = dx5 + dilation_w * v1;
                    int x16 = dx6 + dilation_w * v1;
                    int x17 = dx7 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int y12 = dy2 + dilation_h * u1;
                    int y13 = dy3 + dilation_h * u1;
                    int y14 = dy4 + dilation_h * u1;
                    int y15 = dy5 + dilation_h * u1;
                    int y16 = dy6 + dilation_h * u1;
                    int y17 = dy7 + dilation_h * u1;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;
                    const signed char* sptr04 = img0.row<const signed char>(y04) + x04;
                    const signed char* sptr05 = img0.row<const signed char>(y05) + x05;
                    const signed char* sptr06 = img0.row<const signed char>(y06) + x06;
                    const signed char* sptr07 = img0.row<const signed char>(y07) + x07;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;
                    const signed char* sptr14 = img1.row<const signed char>(y14) + x14;
                    const signed char* sptr15 = img1.row<const signed char>(y15) + x15;
                    const signed char* sptr16 = img1.row<const signed char>(y16) + x16;
                    const signed char* sptr17 = img1.row<const signed char>(y17) + x17;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr01[0];
                    pp[3] = sptr11[0];
                    pp[4] = sptr02[0];
                    pp[5] = sptr12[0];
                    pp[6] = sptr03[0];
                    pp[7] = sptr13[0];
                    pp[8] = sptr04[0];
                    pp[9] = sptr14[0];
                    pp[10] = sptr05[0];
                    pp[11] = sptr15[0];
                    pp[12] = sptr06[0];
                    pp[13] = sptr16[0];
                    pp[14] = sptr07[0];
                    pp[15] = sptr17[0];
                    pp += 16;
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int x1 = dx1 + dilation_w * v;
                int x2 = dx2 + dilation_w * v;
                int x3 = dx3 + dilation_w * v;
                int x4 = dx4 + dilation_w * v;
                int x5 = dx5 + dilation_w * v;
                int x6 = dx6 + dilation_w * v;
                int x7 = dx7 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;
                int y1 = dy1 + dilation_h * u;
                int y2 = dy2 + dilation_h * u;
                int y3 = dy3 + dilation_h * u;
                int y4 = dy4 + dilation_h * u;
                int y5 = dy5 + dilation_h * u;
                int y6 = dy6 + dilation_h * u;
                int y7 = dy7 + dilation_h * u;

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;
                const signed char* sptr2 = img.row<const signed char>(y2) + x2 * elempack;
                const signed char* sptr3 = img.row<const signed char>(y3) + x3 * elempack;
                const signed char* sptr4 = img.row<const signed char>(y4) + x4 * elempack;
                const signed char* sptr5 = img.row<const signed char>(y5) + x5 * elempack;
                const signed char* sptr6 = img.row<const signed char>(y6) + x6 * elempack;
                const signed char* sptr7 = img.row<const signed char>(y7) + x7 * elempack;

                if (elempack == 8)
                {
                    int16x4_t _r0 = vreinterpret_s16_s8(vld1_s8(sptr0));
                    int16x4_t _r1 = vreinterpret_s16_s8(vld1_s8(sptr1));
                    int16x4_t _r2 = vreinterpret_s16_s8(vld1_s8(sptr2));
                    int16x4_t _r3 = vreinterpret_s16_s8(vld1_s8(sptr3));
                    int16x4_t _r4 = vreinterpret_s16_s8(vld1_s8(sptr4));
                    int16x4_t _r5 = vreinterpret_s16_s8(vld1_s8(sptr5));
                    int16x4_t _r6 = vreinterpret_s16_s8(vld1_s8(sptr6));
                    int16x4_t _r7 = vreinterpret_s16_s8(vld1_s8(sptr7));
                    int16x4x2_t _r01 = vzip_s16(_r0, _r1);
                    int16x4x2_t _r23 = vzip_s16(_r2, _r3);
                    int16x4x2_t _r45 = vzip_s16(_r4, _r5);
                    int16x4x2_t _r67 = vzip_s16(_r6, _r7);
                    int32x4x4_t _r0123;
                    _r0123.val[0] = vreinterpretq_s32_s16(vcombine_s16(_r01.val[0], _r01.val[1]));
                    _r0123.val[1] = vreinterpretq_s32_s16(vcombine_s16(_r23.val[0], _r23.val[1]));
                    _r0123.val[2] = vreinterpretq_s32_s16(vcombine_s16(_r45.val[0], _r45.val[1]));
                    _r0123.val[3] = vreinterpretq_s32_s16(vcombine_s16(_r67.val[0], _r67.val[1]));
                    vst4q_s32((int*)pp, _r0123);
                    pp += 64;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp[4] = sptr4[0];
                    pp[5] = sptr5[0];
                    pp[6] = sptr6[0];
                    pp[7] = sptr7[0];
                    pp += 8;
                }
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        int dy0 = (j + jj) / outw * stride_h;
        int dy1 = (j + jj + 1) / outw * stride_h;
        int dy2 = (j + jj + 2) / outw * stride_h;
        int dy3 = (j + jj + 3) / outw * stride_h;
        int dx0 = (j + jj) % outw * stride_w;
        int dx1 = (j + jj + 1) % outw * stride_w;
        int dx2 = (j + jj + 2) % outw * stride_w;
        int dx3 = (j + jj + 3) % outw * stride_w;

        if (dy0 == dy3)
        {
            int kk = 0;
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    if (stride_w == 1)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r01 = vzip_s8(_r0, _r1).val[0];
                        vst1_s8(pp, _r01);
                        pp += 8;
                    }
                    else if (stride_w == 2)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r01 = vtrn_s8(_r0, _r1).val[0];
                        vst1_s8(pp, _r01);
                        pp += 8;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr0[stride_w];
                        pp[3] = sptr1[stride_w];
                        pp[4] = sptr0[stride_w * 2];
                        pp[5] = sptr1[stride_w * 2];
                        pp[6] = sptr0[stride_w * 3];
                        pp[7] = sptr1[stride_w * 3];
                        pp += 8;
                    }
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;

                const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

                if (elempack == 8)
                {
                    int16x4x4_t _r0123;
                    _r0123.val[0] = vreinterpret_s16_s8(vld1_s8(sptr));
                    _r0123.val[1] = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 8));
                    _r0123.val[2] = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 16));
                    _r0123.val[3] = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 24));
                    vst4_s16((short*)pp, _r0123);
                    pp += 32;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp[2] = sptr[stride_w * 2];
                    pp[3] = sptr[stride_w * 3];
                    pp += 4;
                }
            }
        }
        else
        {
            int kk = 0;
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int x02 = dx2 + dilation_w * v0;
                    int x03 = dx3 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int y02 = dy2 + dilation_h * u0;
                    int y03 = dy3 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int x12 = dx2 + dilation_w * v1;
                    int x13 = dx3 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int y12 = dy2 + dilation_h * u1;
                    int y13 = dy3 + dilation_h * u1;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr01[0];
                    pp[3] = sptr11[0];
                    pp[4] = sptr02[0];
                    pp[5] = sptr12[0];
                    pp[6] = sptr03[0];
                    pp[7] = sptr13[0];
                    pp += 8;
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int x1 = dx1 + dilation_w * v;
                int x2 = dx2 + dilation_w * v;
                int x3 = dx3 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;
                int y1 = dy1 + dilation_h * u;
                int y2 = dy2 + dilation_h * u;
                int y3 = dy3 + dilation_h * u;

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;
                const signed char* sptr2 = img.row<const signed char>(y2) + x2 * elempack;
                const signed char* sptr3 = img.row<const signed char>(y3) + x3 * elempack;

                if (elempack == 8)
                {
                    int16x4x4_t _r0123;
                    _r0123.val[0] = vreinterpret_s16_s8(vld1_s8(sptr0));
                    _r0123.val[1] = vreinterpret_s16_s8(vld1_s8(sptr1));
                    _r0123.val[2] = vreinterpret_s16_s8(vld1_s8(sptr2));
                    _r0123.val[3] = vreinterpret_s16_s8(vld1_s8(sptr3));
                    vst4_s16((short*)pp, _r0123);
                    pp += 32;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp += 4;
                }
            }
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
        int dy0 = (j + jj) / outw * stride_h;
        int dy1 = (j + jj + 1) / outw * stride_h;
        int dx0 = (j + jj) % outw * stride_w;
        int dx1 = (j + jj + 1) % outw * stride_w;

        if (dy0 == dy1)
        {
            int kk = 0;
#if __riscv_vector
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr0[stride_w];
                    pp[3] = sptr1[stride_w];
                    pp += 4;
                }
            }
#endif // __riscv_vector
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;

                const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

#if __riscv_vector
                if (elempack == 8)
                {
                    int16x4x2_t _r01;
                    _r01.val[0] = vreinterpret_s16_s8(vld1_s8(sptr));
                    _r01.val[1] = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 8));
                    vst2_s16((short*)pp, _r01);
                    pp += 16;
                }
#endif // __riscv_vector
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp += 2;
                }
            }
        }
        else
        {
            int kk = 0;
#if __riscv_vector
            if (elempack == 1)
            {
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr01[0];
                    pp[3] = sptr11[0];
                    pp += 4;
                }
            }
#endif // __riscv_vector
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int x1 = dx1 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;
                int y1 = dy1 + dilation_h * u;

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;

#if __riscv_vector
                if (elempack == 8)
                {
                    int16x4x2_t _r01;
                    _r01.val[0] = vreinterpret_s16_s8(vld1_s8(sptr0));
                    _r01.val[1] = vreinterpret_s16_s8(vld1_s8(sptr1));
                    vst2_s16((short*)pp, _r01);
                    pp += 16;
                }
#endif // __riscv_vector
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp += 2;
                }
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
        int dy = (j + jj) / outw * stride_h;
        int dx = (j + jj) % outw * stride_w;

        int kk = 0;
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x = dx + dilation_w * v;
            int y = dy + dilation_h * u;

            const signed char* sptr = img.row<const signed char>(y) + x * elempack;

#if __riscv_vector
            if (elempack == 8)
            {
                vst1_s8(pp, vld1_s8(sptr));
                pp += 8;
            }
#endif // __riscv_vector
            if (elempack == 1)
            {
                pp[0] = sptr[0];
                pp += 1;
            }
        }
    }
}

static void convolution_im2col_gemm_transform_kernel_int8(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    // NCNN_LOGE("convolution_im2col_gemm_transform_kernel");
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_int8(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    int elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        elempack = inch % 8 == 0 ? 8 : 1;
    }
#endif // __riscv_vector

    // maxk-inch-outch to pa-maxk-inch/pa-outch
    Mat A_data;
    if (maxk == 1)
    {
        A_data = kernel.reshape(maxk * inch, outch);
    }
    else
    {
        Mat weight_data_r2 = kernel.reshape(maxk, inch, outch);

        A_data.create(maxk * inch, outch, (size_t)1u, 1);

        for (int q = 0; q < outch; q += 1)
        {
            signed char* g00 = A_data.row<signed char>(q);

            for (int p = 0; p + (elempack - 1) < inch; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        const signed char* k00 = weight_data_r2.channel(q).row<const signed char>(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }

    AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, (size_t)1u, 1);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            convolution_im2col_pack_A_tile_int8(A_data, AT_tile, i, max_ii, k, max_kk);
        }
    }
}

static void convolution_im2col_gemm_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_int8(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 1u, opt.workspace_allocator);

    const int nn_NK = nn_N * nn_K;

    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min((N - j), TILE_N);
        const int max_kk = std::min((K - k), TILE_K);

        Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

        // im2col
        convolution_im2col_input_tile_int8(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
    }

    Mat topT_tileX;
    if (K > TILE_K)
        topT_tileX.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat topT_tile;
        if (K > TILE_K)
            topT_tile = topT_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                const Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = k + TILE_K >= K;

                convolution_gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end);
            }
        }
    }
}
