// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 Xinyu302 Limited. All rights reserved.
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

static void convdw3x3s1_pack8_int8_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    int vl = csrr_vlenb() / 2;

    const int group = bottom_blob.c;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        const signed char* k0 = kernel.row<const signed char>(g);

        int* outptr0 = out.row<int>(0);
        int* outptr1 = out.row<int>(1);

        const Mat img0 = bottom_blob.channel(g);

        const signed char* r0 = img0.row<const signed char>(0);
        const signed char* r1 = img0.row<const signed char>(1);
        const signed char* r2 = img0.row<const signed char>(2);
        const signed char* r3 = img0.row<const signed char>(3);

        vl = 8;
        vint8m1_t _k00 = vle8_v_i8m1(k0, vl);
        vint8m1_t _k01 = vle8_v_i8m1(k0 + 8, vl);
        vint8m1_t _k02 = vle8_v_i8m1(k0 + 16, vl);
        vint8m1_t _k10 = vle8_v_i8m1(k0 + 24, vl);
        vint8m1_t _k11 = vle8_v_i8m1(k0 + 32, vl);
        vint8m1_t _k12 = vle8_v_i8m1(k0 + 40, vl);
        vint8m1_t _k20 = vle8_v_i8m1(k0 + 48, vl);
        vint8m1_t _k21 = vle8_v_i8m1(k0 + 56, vl);
        vint8m1_t _k22 = vle8_v_i8m1(k0 + 64, vl);

        // int8x8_t _k00 = vld1_s8(k0);
        // int8x8_t _k01 = vld1_s8(k0 + 8);
        // int8x8_t _k02 = vld1_s8(k0 + 16);
        // int8x8_t _k10 = vld1_s8(k0 + 24);
        // int8x8_t _k11 = vld1_s8(k0 + 32);
        // int8x8_t _k12 = vld1_s8(k0 + 40);
        // int8x8_t _k20 = vld1_s8(k0 + 48);
        // int8x8_t _k21 = vld1_s8(k0 + 56);
        // int8x8_t _k22 = vld1_s8(k0 + 64);

        int i = 0;
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j < outw; j++)
            {
                vint8m1_t _r00 = vle8_v_i8m1(r0, vl);
                vint8m1_t _r01 = vle8_v_i8m1(r0 + 8, vl);
                vint8m1_t _r02 = vle8_v_i8m1(r0 + 16, vl);
                vint8m1_t _r10 = vle8_v_i8m1(r1, vl);
                vint8m1_t _r11 = vle8_v_i8m1(r1 + 8, vl);
                vint8m1_t _r12 = vle8_v_i8m1(r1 + 16, vl);
                vint8m1_t _r20 = vle8_v_i8m1(r2, vl);
                vint8m1_t _r21 = vle8_v_i8m1(r2 + 8, vl);
                vint8m1_t _r22 = vle8_v_i8m1(r2 + 16, vl);

                // int8x8_t _r00 = vld1_s8(r0);
                // int8x8_t _r01 = vld1_s8(r0 + 8);
                // int8x8_t _r02 = vld1_s8(r0 + 16);
                // int8x8_t _r10 = vld1_s8(r1);
                // int8x8_t _r11 = vld1_s8(r1 + 8);
                // int8x8_t _r12 = vld1_s8(r1 + 16);
                // int8x8_t _r20 = vld1_s8(r2);
                // int8x8_t _r21 = vld1_s8(r2 + 8);
                // int8x8_t _r22 = vld1_s8(r2 + 16);

                vint16m2_t _s0 = vwmul_vv_i16m2(_r00, _k00, vl);
                vint16m2_t _s1 = vwmul_vv_i16m2(_r01, _k01, vl);
                vint16m2_t _s2 = vwmul_vv_i16m2(_r02, _k02, vl);
                vint16m2_t _s3 = vwmul_vv_i16m2(_r10, _k10, vl);

                // int16x8_t _s0 = vmull_s8(_r00, _k00);
                // int16x8_t _s1 = vmull_s8(_r01, _k01);
                // int16x8_t _s2 = vmull_s8(_r02, _k02);
                // int16x8_t _s3 = vmull_s8(_r10, _k10);

                _s0 = vwmacc_vv_i16m2(_s0, _r11, _k11, vl);
                _s1 = vwmacc_vv_i16m2(_s1, _r12, _k12, vl);
                _s2 = vwmacc_vv_i16m2(_s2, _r20, _k20, vl);
                _s3 = vwmacc_vv_i16m2(_s3, _r21, _k21, vl);

                // _s0 = vmlal_s8(_s0, _r11, _k11);
                // _s1 = vmlal_s8(_s1, _r12, _k12);
                // _s2 = vmlal_s8(_s2, _r20, _k20);
                // _s3 = vmlal_s8(_s3, _r21, _k21);

                vint16m2_t _s4 = vwmul_vv_i16m2(_r22, _k22, vl);
                // int16x8_t _s4 = vmull_s8(_r22, _k22);

                vint16m1_t _s0_m1 = vget_v_i16m2_i16m1(_s0, 0);
                vint16m1_t _s1_m1 = vget_v_i16m2_i16m1(_s1, 0);
                vint16m1_t _s2_m1 = vget_v_i16m2_i16m1(_s2, 0);
                vint16m1_t _s3_m1 = vget_v_i16m2_i16m1(_s3, 0);
                vint16m1_t _s4_m1 = vget_v_i16m2_i16m1(_s4, 0);

                vint32m2_t _sum = vwadd_vv_i32m2(_s0_m1, _s1_m1, vl);
                _sum = vwadd_wv_i32m2(_sum, _s2_m1, vl);
                _sum = vwadd_wv_i32m2(_sum, _s3_m1, vl);
                _sum = vwadd_wv_i32m2(_sum, _s4_m1, vl);

                // int32x4_t _sum0 = vaddl_s16(vget_low_s16(_s0), vget_low_s16(_s1));
                // int32x4_t _sum1 = vaddl_s16(vget_high_s16(_s0), vget_high_s16(_s1));
                // int32x4_t _sum2 = vaddl_s16(vget_low_s16(_s2), vget_low_s16(_s3));
                // int32x4_t _sum3 = vaddl_s16(vget_high_s16(_s2), vget_high_s16(_s3));
                // _sum0 = vaddw_s16(_sum0, vget_low_s16(_s4));
                // _sum1 = vaddw_s16(_sum1, vget_high_s16(_s4));
                // _sum0 = vaddq_s32(_sum0, _sum2);
                // _sum1 = vaddq_s32(_sum1, _sum3);

                vse32_v_i32m2(outptr0, _sum, vl);
                // vst1q_s32(outptr0, _sum0);
                // vst1q_s32(outptr0 + 4, _sum1);
                r0 += 8;
                r1 += 8;
                r2 += 8;
                outptr0 += 8;
            }

            r0 += 2 * 8;
            r1 += 2 * 8;
            r2 += 2 * 8;
        }
    }
}

static void convdw3x3s2_pack8_int8_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int vl = csrr_vlenb() / 2;

    const int group = bottom_blob.c;

    const int tailstep = (w - 2 * outw + w) * 8;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        const signed char* k0 = kernel.row<const signed char>(g);

        int* outptr0 = out;

        const Mat img0 = bottom_blob.channel(g);

        const signed char* r0 = img0.row<const signed char>(0);
        const signed char* r1 = img0.row<const signed char>(1);
        const signed char* r2 = img0.row<const signed char>(2);

        vint8m1_t _k00 = vle8_v_i8m1(k0, vl);
        vint8m1_t _k01 = vle8_v_i8m1(k0 + 8, vl);
        vint8m1_t _k02 = vle8_v_i8m1(k0 + 16, vl);
        vint8m1_t _k10 = vle8_v_i8m1(k0 + 24, vl);
        vint8m1_t _k11 = vle8_v_i8m1(k0 + 32, vl);
        vint8m1_t _k12 = vle8_v_i8m1(k0 + 40, vl);
        vint8m1_t _k20 = vle8_v_i8m1(k0 + 48, vl);
        vint8m1_t _k21 = vle8_v_i8m1(k0 + 56, vl);
        vint8m1_t _k22 = vle8_v_i8m1(k0 + 64, vl);

        // int8x8_t _k00 = vld1_s8(k0);
        // int8x8_t _k01 = vld1_s8(k0 + 8);
        // int8x8_t _k02 = vld1_s8(k0 + 16);
        // int8x8_t _k10 = vld1_s8(k0 + 24);
        // int8x8_t _k11 = vld1_s8(k0 + 32);
        // int8x8_t _k12 = vld1_s8(k0 + 40);
        // int8x8_t _k20 = vld1_s8(k0 + 48);
        // int8x8_t _k21 = vld1_s8(k0 + 56);
        // int8x8_t _k22 = vld1_s8(k0 + 64);

        int i = 0;
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j < outw; j++)
            {
                vint8m1_t _r00 = vle8_v_i8m1(r0, vl);
                vint8m1_t _r01 = vle8_v_i8m1(r0 + 8, vl);
                vint8m1_t _r02 = vle8_v_i8m1(r0 + 16, vl);
                vint8m1_t _r10 = vle8_v_i8m1(r1, vl);
                vint8m1_t _r11 = vle8_v_i8m1(r1 + 8, vl);
                vint8m1_t _r12 = vle8_v_i8m1(r1 + 16, vl);
                vint8m1_t _r20 = vle8_v_i8m1(r2, vl);
                vint8m1_t _r21 = vle8_v_i8m1(r2 + 8, vl);
                vint8m1_t _r22 = vle8_v_i8m1(r2 + 16, vl);

                // int8x8_t _r00 = vld1_s8(r0);
                // int8x8_t _r01 = vld1_s8(r0 + 8);
                // int8x8_t _r02 = vld1_s8(r0 + 16);
                // int8x8_t _r10 = vld1_s8(r1);
                // int8x8_t _r11 = vld1_s8(r1 + 8);
                // int8x8_t _r12 = vld1_s8(r1 + 16);
                // int8x8_t _r20 = vld1_s8(r2);
                // int8x8_t _r21 = vld1_s8(r2 + 8);
                // int8x8_t _r22 = vld1_s8(r2 + 16);

                vint16m2_t _s0 = vwmul_vv_i16m2(_r00, _k00, vl);
                vint16m2_t _s1 = vwmul_vv_i16m2(_r01, _k01, vl);
                vint16m2_t _s2 = vwmul_vv_i16m2(_r02, _k02, vl);
                vint16m2_t _s3 = vwmul_vv_i16m2(_r10, _k10, vl);

                _s0 = vwmacc_vv_i16m2(_s0, _r11, _k11, vl);
                _s1 = vwmacc_vv_i16m2(_s1, _r12, _k12, vl);
                _s2 = vwmacc_vv_i16m2(_s2, _r20, _k20, vl);
                _s3 = vwmacc_vv_i16m2(_s3, _r21, _k21, vl);

                vint16m2_t _s4 = vwmul_vv_i16m2(_r22, _k22, vl);

                // int16x8_t _s0 = vmull_s8(_r00, _k00);
                // int16x8_t _s1 = vmull_s8(_r01, _k01);
                // int16x8_t _s2 = vmull_s8(_r02, _k02);
                // int16x8_t _s3 = vmull_s8(_r10, _k10);
                // _s0 = vmlal_s8(_s0, _r11, _k11);
                // _s1 = vmlal_s8(_s1, _r12, _k12);
                // _s2 = vmlal_s8(_s2, _r20, _k20);
                // _s3 = vmlal_s8(_s3, _r21, _k21);
                // int16x8_t _s4 = vmull_s8(_r22, _k22);

                vint16m1_t _s0_m1 = vget_v_i16m2_i16m1(_s0, 0);
                vint16m1_t _s1_m1 = vget_v_i16m2_i16m1(_s1, 0);
                vint16m1_t _s2_m1 = vget_v_i16m2_i16m1(_s2, 0);
                vint16m1_t _s3_m1 = vget_v_i16m2_i16m1(_s3, 0);
                vint16m1_t _s4_m1 = vget_v_i16m2_i16m1(_s4, 0);

                vint32m2_t _sum = vwadd_vv_i32m2(_s0_m1, _s1_m1, vl);
                _sum = vwadd_wv_i32m2(_sum, _s2_m1, vl);
                _sum = vwadd_wv_i32m2(_sum, _s3_m1, vl);
                _sum = vwadd_wv_i32m2(_sum, _s4_m1, vl);

                vse32_v_i32m2(outptr0, _sum, vl);

                // int32x4_t _sum0 = vaddl_s16(vget_low_s16(_s0), vget_low_s16(_s1));
                // int32x4_t _sum1 = vaddl_s16(vget_high_s16(_s0), vget_high_s16(_s1));
                // int32x4_t _sum2 = vaddl_s16(vget_low_s16(_s2), vget_low_s16(_s3));
                // int32x4_t _sum3 = vaddl_s16(vget_high_s16(_s2), vget_high_s16(_s3));
                // _sum0 = vaddw_s16(_sum0, vget_low_s16(_s4));
                // _sum1 = vaddw_s16(_sum1, vget_high_s16(_s4));
                // _sum0 = vaddq_s32(_sum0, _sum2);
                // _sum1 = vaddq_s32(_sum1, _sum3);

                // vst1q_s32(outptr0, _sum0);
                // vst1q_s32(outptr0 + 4, _sum1);

                r0 += 16;
                r1 += 16;
                r2 += 16;
                outptr0 += 8;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}
