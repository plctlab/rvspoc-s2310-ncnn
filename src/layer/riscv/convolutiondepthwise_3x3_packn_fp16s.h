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

#define RVV_ASM_FTY

static void convdw3x3s1_packn_fp16sa_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        vfloat16m1_t _bias0 = bias ? vle16_v_f16m1(bias + g * packn, vl) : vfmv_v_f_f16m1((__fp16)0.f, vl);

        const __fp16* k0 = kernel.row<const __fp16>(g);

        __fp16* outptr0 = out.row<__fp16>(0);
        __fp16* outptr1 = out.row<__fp16>(1);

        const Mat img0 = bottom_blob.channel(g);

        const __fp16* r0 = img0.row<const __fp16>(0);
        const __fp16* r1 = img0.row<const __fp16>(1);
        const __fp16* r2 = img0.row<const __fp16>(2);
        const __fp16* r3 = img0.row<const __fp16>(3);

        vfloat16m1_t _k00 = vle16_v_f16m1(k0, vl);
        vfloat16m1_t _k01 = vle16_v_f16m1(k0 + packn, vl);
        vfloat16m1_t _k02 = vle16_v_f16m1(k0 + packn * 2, vl);
        vfloat16m1_t _k10 = vle16_v_f16m1(k0 + packn * 3, vl);
        vfloat16m1_t _k11 = vle16_v_f16m1(k0 + packn * 4, vl);
        vfloat16m1_t _k12 = vle16_v_f16m1(k0 + packn * 5, vl);
        vfloat16m1_t _k20 = vle16_v_f16m1(k0 + packn * 6, vl);
        vfloat16m1_t _k21 = vle16_v_f16m1(k0 + packn * 7, vl);
        vfloat16m1_t _k22 = vle16_v_f16m1(k0 + packn * 8, vl);

        int i = 0;
        for (; i + 1 < outh; i += 2)
        {
            int j = 0;
            for (; j + 1 < outw; j += 2)
            {
#ifdef RVV_ASM_FTY
                asm volatile(
                    "vle.v          v4,             (%[r0])             \n\t"
                    "mv             t1,             %[packn]            \n\t"
                    "slli           t1,             t1,             1   \n\t" // shift left 1 (* 2)
                    "add            %[r0],          %[r0],          t1  \n\t"
                    "vle.v          v5,             (%[r0])             \n\t"
                    "add            %[r0],          %[r0],          t1  \n\t"
                    "vmv.v.v        v0,             %[_bias0]           \n\t"
                    "vle.v          v6,             (%[r0])             \n\t"
                    "add            %[r0],          %[r0],          t1  \n\t"
                    "vmv.v.v        v1,             %[_bias0]           \n\t"
                    "vle.v          v7,             (%[r0])             \n\t"
                    "sub            %[r0],          %[r0],          t1  \n\t"
                    "vmv.v.v        v2,             %[_bias0]           \n\t"
                    "vmv.v.v        v3,             %[_bias0]           \n\t"
                    "vfmacc.vv      v0,             %[_k00],        v4  \n\t"
                    "vle.v          v8,             (%[r1])             \n\t"
                    "add            %[r1],          %[r1],          t1  \n\t"
                    "vfmacc.vv      v0,             %[_k01],        v5  \n\t"
                    "vle.v          v9,             (%[r1])             \n\t"
                    "add            %[r1],          %[r1],          t1  \n\t"
                    "vfmacc.vv      v0,             %[_k02],        v6  \n\t"
                    "vle.v          v10,             (%[r1])             \n\t"
                    "add            %[r1],          %[r1],          t1  \n\t"
                    "vfmacc.vv      v1,             %[_k00],        v5  \n\t"
                    "vle.v          v11,            (%[r1])             \n\t"
                    "sub            %[r1],          %[r1],          t1  \n\t"
                    "vfmacc.vv      v1,             %[_k01],        v6  \n\t"
                    "vle.v          v12,            (%[r2])             \n\t"
                    "add            %[r2],          %[r2],          t1  \n\t"
                    "vfmacc.vv      v1,             %[_k02],        v7  \n\t"
                    "vle.v          v13,            (%[r2])             \n\t"
                    "add            %[r2],          %[r2],          t1  \n\t"
                    "vfmacc.vv      v0,             %[_k10],        v8  \n\t"
                    "vle.v          v14,            (%[r2])             \n\t"
                    "add            %[r2],          %[r2],          t1  \n\t"
                    "vfmacc.vv      v0,             %[_k11],        v9  \n\t"
                    "vle.v          v15,            (%[r2])             \n\t"
                    "sub            %[r2],          %[r2],          t1  \n\t"
                    "vfmacc.vv      v0,             %[_k12],        v10 \n\t"
                    "vle.v          v16,            (%[r3])             \n\t"
                    "add            %[r3],          %[r3],          t1  \n\t"
                    "vfmacc.vv      v1,             %[_k10],        v9  \n\t"
                    "vle.v          v17,            (%[r3])             \n\t"
                    "add            %[r3],          %[r3],          t1  \n\t"
                    "vfmacc.vv      v1,             %[_k11],        v10 \n\t"
                    "vle.v          v18,            (%[r3])             \n\t"
                    "add            %[r3],          %[r3],          t1  \n\t"
                    "vfmacc.vv      v1,             %[_k12],        v11 \n\t"
                    "vle.v          v19,            (%[r3])             \n\t"
                    "sub            %[r3],          %[r3],          t1  \n\t"
                    "vfmacc.vv      v0,             %[_k20],        v12 \n\t"
                    "vfmacc.vv      v0,             %[_k21],        v13 \n\t"
                    "vfmacc.vv      v0,             %[_k22],        v14 \n\t"
                    "vfmacc.vv      v1,             %[_k20],        v13 \n\t"
                    "vfmacc.vv      v1,             %[_k21],        v14 \n\t"
                    "vfmacc.vv      v1,             %[_k22],        v15 \n\t"
                    "vse.v          v0,             (%[outptr0])        \n\t"
                    "add            %[outptr0],     %[outptr0],     t1  \n\t"
                    "vfmacc.vv      v2,             %[_k00],        v8  \n\t"
                    "vfmacc.vv      v2,             %[_k01],        v9  \n\t"
                    "vfmacc.vv      v2,             %[_k02],        v10 \n\t"
                    "vse.v          v1,             (%[outptr0])        \n\t"
                    "add            %[outptr0],     %[outptr0],     t1  \n\t"
                    "vfmacc.vv      v2,             %[_k10],        v12 \n\t"
                    "vfmacc.vv      v2,             %[_k11],        v13 \n\t"
                    "vfmacc.vv      v2,             %[_k12],        v14 \n\t"
                    "vfmacc.vv      v2,             %[_k20],        v16 \n\t"
                    "vfmacc.vv      v2,             %[_k21],        v17 \n\t"
                    "vfmacc.vv      v2,             %[_k22],        v18 \n\t"
                    "vfmacc.vv      v3,             %[_k00],        v9  \n\t"
                    "vfmacc.vv      v3,             %[_k01],        v10 \n\t"
                    "vfmacc.vv      v3,             %[_k02],        v11 \n\t"
                    "vse.v          v2,             (%[outptr1])        \n\t"
                    "vfmacc.vv      v3,             %[_k10],        v13 \n\t"
                    "vfmacc.vv      v3,             %[_k11],        v14 \n\t"
                    "vfmacc.vv      v3,             %[_k12],        v15 \n\t"
                    "vfmacc.vv      v3,             %[_k20],        v17 \n\t"
                    "vfmacc.vv      v3,             %[_k21],        v18 \n\t"
                    "vfmacc.vv      v3,             %[_k22],        v19 \n\t"
                    "add            %[outptr1],     %[outptr1],     t1  \n\t"
                    "vse.v          v3,             (%[outptr1])        \n\t"
                    "add            %[outptr1],     %[outptr1],     t1  \n\t"

                    : [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), [r3] "+r"(r3),
                    [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1)
                    : [packn] "r"(packn), [_bias0] "vr"(_bias0),
                    [_k00] "vr"(_k00), [_k01] "vr"(_k01), [_k02] "vr"(_k02),
                    [_k10] "vr"(_k10), [_k11] "vr"(_k11), [_k12] "vr"(_k12),
                    [_k20] "vr"(_k20), [_k21] "vr"(_k21), [_k22] "vr"(_k22)
                    : "cc", "memory",
                    "t1",
                    "v0", "v1", "v2", "v3",     // sum00, sum01, sum10, sum11
                    "v4", "v5", "v6", "v7",     // r00, r01, r02, r03
                    "v8", "v9", "v10", "v11",   // r10, r11, r12, r13
                    "v12", "v13", "v14", "v15", // r20, r21, r22, r23
                    "v16", "v17", "v18", "v19"  // r30, r31, r32, r33
                );
#else
                vfloat16m1_t _sum00 = _bias0;
                vfloat16m1_t _sum01 = _bias0;
                vfloat16m1_t _sum10 = _bias0;
                vfloat16m1_t _sum11 = _bias0;

                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + packn, vl);
                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + packn * 2, vl);
                vfloat16m1_t _r03 = vle16_v_f16m1(r0 + packn * 3, vl);

                _sum00 = vfmacc_vv_f16m1(_sum00, _k00, _r00, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k01, _r01, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k02, _r02, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k00, _r01, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k01, _r02, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k02, _r03, vl);

                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + packn, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + packn * 2, vl);
                vfloat16m1_t _r13 = vle16_v_f16m1(r1 + packn * 3, vl);

                _sum00 = vfmacc_vv_f16m1(_sum00, _k10, _r10, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k11, _r11, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k12, _r12, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k10, _r11, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k11, _r12, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k12, _r13, vl);
                _sum10 = vfmacc_vv_f16m1(_sum10, _k00, _r10, vl);
                _sum10 = vfmacc_vv_f16m1(_sum10, _k01, _r11, vl);
                _sum10 = vfmacc_vv_f16m1(_sum10, _k02, _r12, vl);
                _sum11 = vfmacc_vv_f16m1(_sum11, _k00, _r11, vl);
                _sum11 = vfmacc_vv_f16m1(_sum11, _k01, _r12, vl);
                _sum11 = vfmacc_vv_f16m1(_sum11, _k02, _r13, vl);

                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + packn, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + packn * 2, vl);
                vfloat16m1_t _r23 = vle16_v_f16m1(r2 + packn * 3, vl);

                _sum00 = vfmacc_vv_f16m1(_sum00, _k20, _r20, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k21, _r21, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k22, _r22, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k20, _r21, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k21, _r22, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k22, _r23, vl);
                _sum10 = vfmacc_vv_f16m1(_sum10, _k10, _r20, vl);
                _sum10 = vfmacc_vv_f16m1(_sum10, _k11, _r21, vl);
                _sum10 = vfmacc_vv_f16m1(_sum10, _k12, _r22, vl);
                _sum11 = vfmacc_vv_f16m1(_sum11, _k10, _r21, vl);
                _sum11 = vfmacc_vv_f16m1(_sum11, _k11, _r22, vl);
                _sum11 = vfmacc_vv_f16m1(_sum11, _k12, _r23, vl);

                vfloat16m1_t _r30 = vle16_v_f16m1(r3, vl);
                vfloat16m1_t _r31 = vle16_v_f16m1(r3 + packn, vl);
                vfloat16m1_t _r32 = vle16_v_f16m1(r3 + packn * 2, vl);
                vfloat16m1_t _r33 = vle16_v_f16m1(r3 + packn * 3, vl);

                _sum10 = vfmacc_vv_f16m1(_sum10, _k20, _r30, vl);
                _sum10 = vfmacc_vv_f16m1(_sum10, _k21, _r31, vl);
                _sum10 = vfmacc_vv_f16m1(_sum10, _k22, _r32, vl);
                _sum11 = vfmacc_vv_f16m1(_sum11, _k20, _r31, vl);
                _sum11 = vfmacc_vv_f16m1(_sum11, _k21, _r32, vl);
                _sum11 = vfmacc_vv_f16m1(_sum11, _k22, _r33, vl);

                vse16_v_f16m1(outptr0, _sum00, vl);
                vse16_v_f16m1(outptr0 + packn, _sum01, vl);
                vse16_v_f16m1(outptr1, _sum10, vl);
                vse16_v_f16m1(outptr1 + packn, _sum11, vl);

                outptr0 += packn * 2;
                outptr1 += packn * 2;

                r0 += packn * 2;
                r1 += packn * 2;
                r2 += packn * 2;
                r3 += packn * 2;
#endif
            }
            for (; j < outw; j++)
            {
                vfloat16m1_t _sum0 = _bias0;
                vfloat16m1_t _sum1 = _bias0;

                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + packn, vl);
                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + packn * 2, vl);

                _sum0 = vfmacc_vv_f16m1(_sum0, _k00, _r00, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k01, _r01, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k02, _r02, vl);

                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + packn, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + packn * 2, vl);

                _sum0 = vfmacc_vv_f16m1(_sum0, _k10, _r10, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k11, _r11, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k12, _r12, vl);
                _sum1 = vfmacc_vv_f16m1(_sum1, _k00, _r10, vl);
                _sum1 = vfmacc_vv_f16m1(_sum1, _k01, _r11, vl);
                _sum1 = vfmacc_vv_f16m1(_sum1, _k02, _r12, vl);

                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + packn, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + packn * 2, vl);

                _sum0 = vfmacc_vv_f16m1(_sum0, _k20, _r20, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k21, _r21, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k22, _r22, vl);
                _sum1 = vfmacc_vv_f16m1(_sum1, _k10, _r20, vl);
                _sum1 = vfmacc_vv_f16m1(_sum1, _k11, _r21, vl);
                _sum1 = vfmacc_vv_f16m1(_sum1, _k12, _r22, vl);

                vfloat16m1_t _r30 = vle16_v_f16m1(r3, vl);
                vfloat16m1_t _r31 = vle16_v_f16m1(r3 + packn, vl);
                vfloat16m1_t _r32 = vle16_v_f16m1(r3 + packn * 2, vl);

                _sum1 = vfmacc_vv_f16m1(_sum1, _k20, _r30, vl);
                _sum1 = vfmacc_vv_f16m1(_sum1, _k21, _r31, vl);
                _sum1 = vfmacc_vv_f16m1(_sum1, _k22, _r32, vl);

                vse16_v_f16m1(outptr0, _sum0, vl);
                vse16_v_f16m1(outptr1, _sum1, vl);

                outptr0 += packn;
                outptr1 += packn;

                r0 += packn;
                r1 += packn;
                r2 += packn;
                r3 += packn;
            }

            r0 += 2 * packn + w * packn;
            r1 += 2 * packn + w * packn;
            r2 += 2 * packn + w * packn;
            r3 += 2 * packn + w * packn;

            outptr0 += outw * packn;
            outptr1 += outw * packn;
        }
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 1 < outw; j += 2)
            {
                vfloat16m1_t _sum00 = _bias0;
                vfloat16m1_t _sum01 = _bias0;

                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + packn, vl);
                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + packn * 2, vl);
                vfloat16m1_t _r03 = vle16_v_f16m1(r0 + packn * 3, vl);

                _sum00 = vfmacc_vv_f16m1(_sum00, _k00, _r00, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k01, _r01, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k02, _r02, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k00, _r01, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k01, _r02, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k02, _r03, vl);

                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + packn, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + packn * 2, vl);
                vfloat16m1_t _r13 = vle16_v_f16m1(r1 + packn * 3, vl);

                _sum00 = vfmacc_vv_f16m1(_sum00, _k10, _r10, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k11, _r11, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k12, _r12, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k10, _r11, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k11, _r12, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k12, _r13, vl);

                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + packn, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + packn * 2, vl);
                vfloat16m1_t _r23 = vle16_v_f16m1(r2 + packn * 3, vl);

                _sum00 = vfmacc_vv_f16m1(_sum00, _k20, _r20, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k21, _r21, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k22, _r22, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k20, _r21, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k21, _r22, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k22, _r23, vl);

                vse16_v_f16m1(outptr0, _sum00, vl);
                vse16_v_f16m1(outptr0 + packn, _sum01, vl);

                outptr0 += packn * 2;

                r0 += packn * 2;
                r1 += packn * 2;
                r2 += packn * 2;
            }
            for (; j < outw; j++)
            {
                vfloat16m1_t _sum0 = _bias0;

                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + packn, vl);
                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + packn * 2, vl);

                _sum0 = vfmacc_vv_f16m1(_sum0, _k00, _r00, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k01, _r01, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k02, _r02, vl);

                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + packn, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + packn * 2, vl);

                _sum0 = vfmacc_vv_f16m1(_sum0, _k10, _r10, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k11, _r11, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k12, _r12, vl);

                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + packn, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + packn * 2, vl);

                _sum0 = vfmacc_vv_f16m1(_sum0, _k20, _r20, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k21, _r21, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k22, _r22, vl);

                vse16_v_f16m1(outptr0, _sum0, vl);

                outptr0 += packn;

                r0 += packn;
                r1 += packn;
                r2 += packn;
            }

            r0 += 2 * packn;
            r1 += 2 * packn;
            r2 += 2 * packn;
        }
    }
}

static void convdw3x3s2_packn_fp16sa_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const int tailstep = (w - 2 * outw + w) * packn;

    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        vfloat16m1_t _bias0 = bias ? vle16_v_f16m1(bias + g * packn, vl) : vfmv_v_f_f16m1((__fp16)0.f, vl);

        const __fp16* k0 = kernel.row<const __fp16>(g);

        __fp16* outptr0 = out;

        const Mat img0 = bottom_blob.channel(g);

        const __fp16* r0 = img0.row<const __fp16>(0);
        const __fp16* r1 = img0.row<const __fp16>(1);
        const __fp16* r2 = img0.row<const __fp16>(2);

        vfloat16m1_t _k00 = vle16_v_f16m1(k0, vl);
        vfloat16m1_t _k01 = vle16_v_f16m1(k0 + packn, vl);
        vfloat16m1_t _k02 = vle16_v_f16m1(k0 + packn * 2, vl);
        vfloat16m1_t _k10 = vle16_v_f16m1(k0 + packn * 3, vl);
        vfloat16m1_t _k11 = vle16_v_f16m1(k0 + packn * 4, vl);
        vfloat16m1_t _k12 = vle16_v_f16m1(k0 + packn * 5, vl);
        vfloat16m1_t _k20 = vle16_v_f16m1(k0 + packn * 6, vl);
        vfloat16m1_t _k21 = vle16_v_f16m1(k0 + packn * 7, vl);
        vfloat16m1_t _k22 = vle16_v_f16m1(k0 + packn * 8, vl);

        int i = 0;
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 1 < outw; j += 2)
            {
#ifdef RVV_ASM_FTY
                asm volatile(
                    "vle.v          v2,             (%[r0])             \n\t"
                    "mv             t1,             %[packn]            \n\t"
                    "slli           t1,             t1,             1   \n\t" // shift left 1 (* 2)
                    "add            %[r0],          %[r0],          t1  \n\t"
                    "vle.v          v3,             (%[r0])             \n\t"
                    "vmv.v.v        v0,             %[_bias0]           \n\t"
                    "add            %[r0],          %[r0],          t1  \n\t"
                    "vle.v          v4,             (%[r0])             \n\t"
                    "add            %[r0],          %[r0],          t1  \n\t"
                    "vmv.v.v        v1,             %[_bias0]           \n\t"
                    "vle.v          v5,             (%[r0])             \n\t"
                    "add            %[r0],          %[r0],          t1  \n\t"
                    "vle.v          v6,             (%[r0])             \n\t"
                    "vfmacc.vv      v0,             %[_k00],        v2  \n\t"
                    "vle.v          v7,             (%[r1])             \n\t"
                    "vfmacc.vv      v0,             %[_k01],        v3  \n\t"
                    "add            %[r1],          %[r1],          t1  \n\t"
                    "vfmacc.vv      v0,             %[_k02],        v4  \n\t"
                    "vle.v          v8,             (%[r1])             \n\t"
                    "vfmacc.vv      v1,             %[_k00],        v4  \n\t"
                    "add            %[r1],          %[r1],          t1  \n\t"
                    "vfmacc.vv      v1,             %[_k01],        v5  \n\t"
                    "vle.v          v9,             (%[r1])             \n\t"
                    "add            %[r1],          %[r1],          t1  \n\t"
                    "vfmacc.vv      v1,             %[_k02],        v6  \n\t"
                    "vle.v          v10,            (%[r1])             \n\t"
                    "add            %[r1],          %[r1],          t1  \n\t"
                    "vfmacc.vv      v0,             %[_k10],        v7  \n\t"
                    "vle.v          v11,            (%[r1])             \n\t"
                    "vfmacc.vv      v0,             %[_k11],        v8  \n\t"
                    "vle.v          v12,            (%[r2])             \n\t"
                    "add            %[r2],          %[r2],          t1  \n\t"
                    "vfmacc.vv      v0,             %[_k12],        v9  \n\t"
                    "vle.v          v13,            (%[r2])             \n\t"
                    "add            %[r2],          %[r2],          t1  \n\t"
                    "vfmacc.vv      v1,             %[_k10],        v9  \n\t"
                    "vle.v          v14,            (%[r2])             \n\t"
                    "add            %[r2],          %[r2],          t1  \n\t"
                    "vfmacc.vv      v0,             %[_k20],        v12 \n\t"
                    "vle.v          v15,            (%[r2])             \n\t"
                    "add            %[r2],          %[r2],          t1  \n\t"
                    "vfmacc.vv      v0,             %[_k21],        v13 \n\t"
                    "vle.v          v16,            (%[r2])             \n\t"
                    "vfmacc.vv      v0,             %[_k22],        v14 \n\t"
                    "vfmacc.vv      v1,             %[_k11],        v10 \n\t"
                    "vfmacc.vv      v1,             %[_k12],        v11 \n\t"
                    "vse.v          v0,             (%[outptr0])        \n\t"
                    "vfmacc.vv      v1,             %[_k20],        v14 \n\t"
                    "vfmacc.vv      v1,             %[_k21],        v15 \n\t"
                    "vfmacc.vv      v1,             %[_k22],        v16 \n\t"
                    "add            %[outptr0],     %[outptr0],     t1  \n\t"
                    "vse.v          v1,             (%[outptr0])        \n\t"
                    "add            %[outptr0],     %[outptr0],     t1  \n\t"
                    : [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), [outptr0] "+r"(outptr0)
                    : [packn] "r"(packn), [_bias0] "vr"(_bias0),
                    [_k00] "vr"(_k00), [_k01] "vr"(_k01), [_k02] "vr"(_k02),
                    [_k10] "vr"(_k10), [_k11] "vr"(_k11), [_k12] "vr"(_k12),
                    [_k20] "vr"(_k20), [_k21] "vr"(_k21), [_k22] "vr"(_k22)
                    : "cc", "memory",
                    "t1",
                    "v0", "v1",                       // sum00, sum01
                    "v2", "v3", "v4", "v5", "v6",     // r00, r01, r02, r03, r04
                    "v7", "v8", "v9", "v10", "v11",   // r10, r11, r12, r13, r14
                    "v12", "v13", "v14", "v15", "v16" // r20, r21, r22, r23, r24
                );
#else
                vfloat16m1_t _sum00 = _bias0;
                vfloat16m1_t _sum01 = _bias0;

                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + packn, vl);
                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + packn * 2, vl);
                vfloat16m1_t _r03 = vle16_v_f16m1(r0 + packn * 3, vl);
                vfloat16m1_t _r04 = vle16_v_f16m1(r0 + packn * 4, vl);

                _sum00 = vfmacc_vv_f16m1(_sum00, _k00, _r00, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k01, _r01, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k02, _r02, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k00, _r02, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k01, _r03, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k02, _r04, vl);

                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + packn, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + packn * 2, vl);
                vfloat16m1_t _r13 = vle16_v_f16m1(r1 + packn * 3, vl);
                vfloat16m1_t _r14 = vle16_v_f16m1(r1 + packn * 4, vl);

                _sum00 = vfmacc_vv_f16m1(_sum00, _k10, _r10, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k11, _r11, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k12, _r12, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k10, _r12, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k11, _r13, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k12, _r14, vl);

                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + packn, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + packn * 2, vl);
                vfloat16m1_t _r23 = vle16_v_f16m1(r2 + packn * 3, vl);
                vfloat16m1_t _r24 = vle16_v_f16m1(r2 + packn * 4, vl);

                _sum00 = vfmacc_vv_f16m1(_sum00, _k20, _r20, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k21, _r21, vl);
                _sum00 = vfmacc_vv_f16m1(_sum00, _k22, _r22, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k20, _r22, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k21, _r23, vl);
                _sum01 = vfmacc_vv_f16m1(_sum01, _k22, _r24, vl);

                vse16_v_f16m1(outptr0, _sum00, vl);
                vse16_v_f16m1(outptr0 + packn, _sum01, vl);

                outptr0 += packn * 2;

                r0 += packn * 4;
                r1 += packn * 4;
                r2 += packn * 4;
#endif
            }
            for (; j < outw; j++)
            {
                vfloat16m1_t _sum0 = _bias0;

                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + packn, vl);
                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + packn * 2, vl);

                _sum0 = vfmacc_vv_f16m1(_sum0, _k00, _r00, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k01, _r01, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k02, _r02, vl);

                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + packn, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + packn * 2, vl);

                _sum0 = vfmacc_vv_f16m1(_sum0, _k10, _r10, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k11, _r11, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k12, _r12, vl);

                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + packn, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + packn * 2, vl);

                _sum0 = vfmacc_vv_f16m1(_sum0, _k20, _r20, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k21, _r21, vl);
                _sum0 = vfmacc_vv_f16m1(_sum0, _k22, _r22, vl);

                vse16_v_f16m1(outptr0, _sum0, vl);

                outptr0 += packn;

                r0 += packn * 2;
                r1 += packn * 2;
                r2 += packn * 2;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}

#undef RVV_ASM_FTY