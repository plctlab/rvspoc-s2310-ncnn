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

static void conv3x3s1_pack1ton_int8_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e8m1(packn);

    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(vmv_v_x_i32m2(0, vl));
        // const int* ptr = top_blob;
        // for (int i = 0; i < top_blob.total(); i++)
        // {
        //     fprintf(stderr, "0x%08x ", ptr[i]);
        //     if (i % 8 == 7)
        //     {
        //         fprintf(stderr, "\n");
        //     }
        // }
        // fprintf(stderr, "\n");

        const int8_t* k0 = kernel.channel(p);

        int q = 0;
        for (; q < inch; q++)
        {
            int32_t* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const int8_t* r0 = img0.row<const int8_t>(0);
            const int8_t* r1 = img0.row<const int8_t>(1);
            const int8_t* r2 = img0.row<const int8_t>(2);

            vint8m1_t _k00 = vle8_v_i8m1(k0 + packn * 0, vl);
            vint8m1_t _k01 = vle8_v_i8m1(k0 + packn * 1, vl);
            vint8m1_t _k02 = vle8_v_i8m1(k0 + packn * 2, vl);
            vint8m1_t _k10 = vle8_v_i8m1(k0 + packn * 3, vl);
            vint8m1_t _k11 = vle8_v_i8m1(k0 + packn * 4, vl);
            vint8m1_t _k12 = vle8_v_i8m1(k0 + packn * 5, vl);
            vint8m1_t _k20 = vle8_v_i8m1(k0 + packn * 6, vl);
            vint8m1_t _k21 = vle8_v_i8m1(k0 + packn * 7, vl);
            vint8m1_t _k22 = vle8_v_i8m1(k0 + packn * 8, vl);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 7 < outw; j += 8)
                {
                    vint32m2_t _sum0 = vle32_v_i32m2(outptr0 + packn * 0, vl);
                    vint32m2_t _sum1 = vle32_v_i32m2(outptr0 + packn * 1, vl);
                    vint32m2_t _sum2 = vle32_v_i32m2(outptr0 + packn * 2, vl);
                    vint32m2_t _sum3 = vle32_v_i32m2(outptr0 + packn * 3, vl);
                    vint32m2_t _sum4 = vle32_v_i32m2(outptr0 + packn * 4, vl);
                    vint32m2_t _sum5 = vle32_v_i32m2(outptr0 + packn * 5, vl);
                    vint32m2_t _sum6 = vle32_v_i32m2(outptr0 + packn * 6, vl);
                    vint32m2_t _sum7 = vle32_v_i32m2(outptr0 + packn * 7, vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[1], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[2], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[3], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[4], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[5], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[6], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[7], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[2], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[3], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[4], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[5], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[6], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[7], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[8], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[3], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[4], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[5], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[6], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[7], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[8], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[9], vl), 0), vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[1], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[2], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[3], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[4], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[5], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[6], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[7], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[2], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[3], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[4], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[5], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[6], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[7], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[8], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[3], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[4], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[5], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[6], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[7], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[8], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[9], vl), 0), vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[1], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[2], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[3], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[4], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[5], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[6], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[7], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[2], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[3], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[4], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[5], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[6], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[7], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[8], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[3], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[4], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[5], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[6], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[7], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[8], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[9], vl), 0), vl);

                    vse32_v_i32m2(outptr0 + packn * 0, _sum0, vl);
                    vse32_v_i32m2(outptr0 + packn * 1, _sum1, vl);
                    vse32_v_i32m2(outptr0 + packn * 2, _sum2, vl);
                    vse32_v_i32m2(outptr0 + packn * 3, _sum3, vl);
                    vse32_v_i32m2(outptr0 + packn * 4, _sum4, vl);
                    vse32_v_i32m2(outptr0 + packn * 5, _sum5, vl);
                    vse32_v_i32m2(outptr0 + packn * 6, _sum6, vl);
                    vse32_v_i32m2(outptr0 + packn * 7, _sum7, vl);

                    outptr0 += packn * 8;

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                }
                for (; j + 3 < outw; j += 4)
                {
                    vint32m2_t _sum0 = vle32_v_i32m2(outptr0 + packn * 0, vl);
                    vint32m2_t _sum1 = vle32_v_i32m2(outptr0 + packn * 1, vl);
                    vint32m2_t _sum2 = vle32_v_i32m2(outptr0 + packn * 2, vl);
                    vint32m2_t _sum3 = vle32_v_i32m2(outptr0 + packn * 3, vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[1], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[2], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[3], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[2], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[3], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[4], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[3], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[4], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[5], vl), 0), vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[1], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[2], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[3], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[2], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[3], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[4], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[3], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[4], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[5], vl), 0), vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[1], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[2], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[3], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[2], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[3], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[4], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[3], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[4], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[5], vl), 0), vl);

                    vse32_v_i32m2(outptr0 + packn * 0, _sum0, vl);
                    vse32_v_i32m2(outptr0 + packn * 1, _sum1, vl);
                    vse32_v_i32m2(outptr0 + packn * 2, _sum2, vl);
                    vse32_v_i32m2(outptr0 + packn * 3, _sum3, vl);

                    outptr0 += packn * 4;

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                }
                for (; j + 1 < outw; j += 2)
                {
                    vint32m2_t _sum0 = vle32_v_i32m2(outptr0 + packn * 0, vl);
                    vint32m2_t _sum1 = vle32_v_i32m2(outptr0 + packn * 1, vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[1], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[2], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[3], vl), 0), vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[1], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[2], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[3], vl), 0), vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[1], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[2], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[3], vl), 0), vl);

                    vse32_v_i32m2(outptr0 + packn * 0, _sum0, vl);
                    vse32_v_i32m2(outptr0 + packn * 1, _sum1, vl);

                    outptr0 += packn * 2;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                }
                for (; j < outw; j++)
                {
                    vint32m2_t _sum0 = vle32_v_i32m2(outptr0 + packn * 0, vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[0], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[1], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[2], vl), 0), vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[0], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[1], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[2], vl), 0), vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[0], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[1], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[2], vl), 0), vl);

                    vse32_v_i32m2(outptr0 + packn * 0, _sum0, vl);

                    outptr0 += packn;

                    r0 += 1;
                    r1 += 1;
                    r2 += 1;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            k0 += 9 * packn;
        }
    }
}

static void conv3x3s2_pack1ton_int8_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e8m1(packn);

    int w = bottom_blob.w;
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(vmv_v_x_i32m2(0, vl));

        const int8_t* k0 = kernel.channel(p);

        int q = 0;
        for (; q < inch; q++)
        {
            int32_t* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const int8_t* r0 = img0.row<const int8_t>(0);
            const int8_t* r1 = img0.row<const int8_t>(1);
            const int8_t* r2 = img0.row<const int8_t>(2);

            vint8m1_t _k00 = vle8_v_i8m1(k0 + packn * 0, vl);
            vint8m1_t _k01 = vle8_v_i8m1(k0 + packn * 1, vl);
            vint8m1_t _k02 = vle8_v_i8m1(k0 + packn * 2, vl);
            vint8m1_t _k10 = vle8_v_i8m1(k0 + packn * 3, vl);
            vint8m1_t _k11 = vle8_v_i8m1(k0 + packn * 4, vl);
            vint8m1_t _k12 = vle8_v_i8m1(k0 + packn * 5, vl);
            vint8m1_t _k20 = vle8_v_i8m1(k0 + packn * 6, vl);
            vint8m1_t _k21 = vle8_v_i8m1(k0 + packn * 7, vl);
            vint8m1_t _k22 = vle8_v_i8m1(k0 + packn * 8, vl);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 7 < outw; j += 8)
                {
                    vint32m2_t _sum0 = vle32_v_i32m2(outptr0 + packn * 0, vl);
                    vint32m2_t _sum1 = vle32_v_i32m2(outptr0 + packn * 1, vl);
                    vint32m2_t _sum2 = vle32_v_i32m2(outptr0 + packn * 2, vl);
                    vint32m2_t _sum3 = vle32_v_i32m2(outptr0 + packn * 3, vl);
                    vint32m2_t _sum4 = vle32_v_i32m2(outptr0 + packn * 4, vl);
                    vint32m2_t _sum5 = vle32_v_i32m2(outptr0 + packn * 5, vl);
                    vint32m2_t _sum6 = vle32_v_i32m2(outptr0 + packn * 6, vl);
                    vint32m2_t _sum7 = vle32_v_i32m2(outptr0 + packn * 7, vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[2], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[4], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[6], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[8], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[10], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[12], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[14], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[3], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[5], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[7], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[9], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[11], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[13], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[15], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[4], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[6], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[8], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[10], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[12], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[14], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[16], vl), 0), vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[2], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[4], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[6], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[8], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[10], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[12], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[14], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[3], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[5], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[7], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[9], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[11], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[13], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[15], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[4], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[6], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[8], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[10], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[12], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[14], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[16], vl), 0), vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[2], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[4], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[6], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[8], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[10], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[12], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[14], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[3], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[5], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[7], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[9], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[11], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[13], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[15], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[4], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[6], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[8], vl), 0), vl);
                    _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[10], vl), 0), vl);
                    _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[12], vl), 0), vl);
                    _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[14], vl), 0), vl);
                    _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[16], vl), 0), vl);

                    vse32_v_i32m2(outptr0 + packn * 0, _sum0, vl);
                    vse32_v_i32m2(outptr0 + packn * 1, _sum1, vl);
                    vse32_v_i32m2(outptr0 + packn * 2, _sum2, vl);
                    vse32_v_i32m2(outptr0 + packn * 3, _sum3, vl);
                    vse32_v_i32m2(outptr0 + packn * 4, _sum4, vl);
                    vse32_v_i32m2(outptr0 + packn * 5, _sum5, vl);
                    vse32_v_i32m2(outptr0 + packn * 6, _sum6, vl);
                    vse32_v_i32m2(outptr0 + packn * 7, _sum7, vl);

                    outptr0 += packn * 8;

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                }
                for (; j + 3 < outw; j += 4)
                {
                    vint32m2_t _sum0 = vle32_v_i32m2(outptr0 + packn * 0, vl);
                    vint32m2_t _sum1 = vle32_v_i32m2(outptr0 + packn * 1, vl);
                    vint32m2_t _sum2 = vle32_v_i32m2(outptr0 + packn * 2, vl);
                    vint32m2_t _sum3 = vle32_v_i32m2(outptr0 + packn * 3, vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[2], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[4], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[6], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[3], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[5], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[7], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[4], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[6], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[8], vl), 0), vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[2], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[4], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[6], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[3], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[5], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[7], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[4], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[6], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[8], vl), 0), vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[2], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[4], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[6], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[3], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[5], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[7], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[4], vl), 0), vl);
                    _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[6], vl), 0), vl);
                    _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[8], vl), 0), vl);

                    vse32_v_i32m2(outptr0 + packn * 0, _sum0, vl);
                    vse32_v_i32m2(outptr0 + packn * 1, _sum1, vl);
                    vse32_v_i32m2(outptr0 + packn * 2, _sum2, vl);
                    vse32_v_i32m2(outptr0 + packn * 3, _sum3, vl);

                    outptr0 += packn * 4;

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                }
                for (; j + 1 < outw; j += 2)
                {
                    vint32m2_t _sum0 = vle32_v_i32m2(outptr0 + packn * 0, vl);
                    vint32m2_t _sum1 = vle32_v_i32m2(outptr0 + packn * 1, vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[2], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[3], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[4], vl), 0), vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[2], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[3], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[4], vl), 0), vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[0], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[2], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[1], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[3], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[2], vl), 0), vl);
                    _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[4], vl), 0), vl);

                    vse32_v_i32m2(outptr0 + packn * 0, _sum0, vl);
                    vse32_v_i32m2(outptr0 + packn * 1, _sum1, vl);

                    outptr0 += packn * 2;

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                }
                for (; j < outw; j++)
                {
                    vint32m2_t _sum0 = vle32_v_i32m2(outptr0 + packn * 0, vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k00, r0[0], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k01, r0[1], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k02, r0[2], vl), 0), vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k10, r1[0], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k11, r1[1], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k12, r1[2], vl), 0), vl);

                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k20, r2[0], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k21, r2[1], vl), 0), vl);
                    _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_k22, r2[2], vl), 0), vl);

                    vse32_v_i32m2(outptr0 + packn * 0, _sum0, vl);

                    outptr0 += packn;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            k0 += 9 * packn;
        }
    }
}
