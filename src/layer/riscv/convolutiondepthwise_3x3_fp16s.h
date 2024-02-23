// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

static void convdw3x3s1_fp16sa_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    size_t vl;

    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const __fp16* kernel = _kernel;
    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        const __fp16 bias0 = bias ? bias[g] : 0.f;

        const __fp16* kernel0 = kernel + g * 9;

        __fp16* outptr0 = out;
        __fp16* outptr1 = outptr0 + outw;
        __fp16* outptr2 = outptr1 + outw;
        __fp16* outptr3 = outptr2 + outw;

        const __fp16* img0 = bottom_blob.channel(g);

        const __fp16* r0 = img0;
        const __fp16* r1 = img0 + w;
        const __fp16* r2 = img0 + w * 2;
        const __fp16* r3 = img0 + w * 3;
        const __fp16* r4 = img0 + w * 4;
        const __fp16* r5 = img0 + w * 5;

        __fp16 _k0 = kernel0[0];
        __fp16 _k1 = kernel0[1];
        __fp16 _k2 = kernel0[2];
        __fp16 _k3 = kernel0[3];
        __fp16 _k4 = kernel0[4];
        __fp16 _k5 = kernel0[5];
        __fp16 _k6 = kernel0[6];
        __fp16 _k7 = kernel0[7];
        __fp16 _k8 = kernel0[8];

        vl = vsetvl_e16m1(packn);
        vfloat16m1_t _bias0 = vfmv_v_f_f16m1(bias0, vl);

        int i = 0;
        for (; i + 3 < outh; i += 4)
        {
            int j = outw;
            while (j > 0)
            {
                vl = vsetvl_e16m1(j);
                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r30 = vle16_v_f16m1(r3, vl);
                vfloat16m1_t _r40 = vle16_v_f16m1(r4, vl);
                vfloat16m1_t _r50 = vle16_v_f16m1(r5, vl);

                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + 1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + 1, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + 1, vl);
                vfloat16m1_t _r31 = vle16_v_f16m1(r3 + 1, vl);
                vfloat16m1_t _r41 = vle16_v_f16m1(r4 + 1, vl);
                vfloat16m1_t _r51 = vle16_v_f16m1(r5 + 1, vl);

                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + 2, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + 2, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + 2, vl);
                vfloat16m1_t _r32 = vle16_v_f16m1(r3 + 2, vl);
                vfloat16m1_t _r42 = vle16_v_f16m1(r4 + 2, vl);
                vfloat16m1_t _r52 = vle16_v_f16m1(r5 + 2, vl);

                vfloat16m1_t _sum0 = _bias0;
                vfloat16m1_t _sum1 = _bias0;
                vfloat16m1_t _sum2 = _bias0;
                vfloat16m1_t _sum3 = _bias0;

                _sum0 = vfmacc_vf_f16m1(_sum0, _k0, _r00, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k1, _r01, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k2, _r02, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k0, _r10, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k1, _r11, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k2, _r12, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, _k0, _r20, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, _k1, _r21, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, _k2, _r22, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, _k0, _r30, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, _k1, _r31, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, _k2, _r32, vl);

                _sum0 = vfmacc_vf_f16m1(_sum0, _k3, _r10, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k4, _r11, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k5, _r12, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k3, _r20, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k4, _r21, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k5, _r22, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, _k3, _r30, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, _k4, _r31, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, _k5, _r32, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, _k3, _r40, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, _k4, _r41, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, _k5, _r42, vl);

                _sum0 = vfmacc_vf_f16m1(_sum0, _k6, _r20, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k7, _r21, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k8, _r22, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k6, _r30, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k7, _r31, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k8, _r32, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, _k6, _r40, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, _k7, _r41, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, _k8, _r42, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, _k6, _r50, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, _k7, _r51, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, _k8, _r52, vl);

                vse16_v_f16m1(outptr0, _sum0, vl);
                vse16_v_f16m1(outptr1, _sum1, vl);
                vse16_v_f16m1(outptr2, _sum2, vl);
                vse16_v_f16m1(outptr3, _sum3, vl);

                r0 += vl;
                r1 += vl;
                r2 += vl;
                r3 += vl;
                r4 += vl;
                r5 += vl;
                outptr0 += vl;
                outptr1 += vl;
                outptr2 += vl;
                outptr3 += vl;
                j -= vl;
            }

            r0 += 2 + 3 * w;
            r1 += 2 + 3 * w;
            r2 += 2 + 3 * w;
            r3 += 2 + 3 * w;
            r4 += 2 + 3 * w;
            r5 += 2 + 3 * w;

            outptr0 += 3 * outw;
            outptr1 += 3 * outw;
            outptr2 += 3 * outw;
            outptr3 += 3 * outw;
        }
        for (; i + 1 < outh; i += 2)
        {
            int j = outw;
            while (j > 0)
            {
                vl = vsetvl_e16m1(j);
                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);
                vfloat16m1_t _r30 = vle16_v_f16m1(r3, vl);

                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + 1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + 1, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + 1, vl);
                vfloat16m1_t _r31 = vle16_v_f16m1(r3 + 1, vl);

                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + 2, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + 2, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + 2, vl);
                vfloat16m1_t _r32 = vle16_v_f16m1(r3 + 2, vl);

                vfloat16m1_t _sum0 = _bias0;
                vfloat16m1_t _sum1 = _bias0;

                _sum0 = vfmacc_vf_f16m1(_sum0, _k0, _r00, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k1, _r01, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k2, _r02, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k0, _r10, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k1, _r11, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k2, _r12, vl);

                _sum0 = vfmacc_vf_f16m1(_sum0, _k3, _r10, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k4, _r11, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k5, _r12, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k3, _r20, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k4, _r21, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k5, _r22, vl);

                _sum0 = vfmacc_vf_f16m1(_sum0, _k6, _r20, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k7, _r21, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k8, _r22, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k6, _r30, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k7, _r31, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, _k8, _r32, vl);

                vse16_v_f16m1(outptr0, _sum0, vl);
                vse16_v_f16m1(outptr1, _sum1, vl);

                r0 += vl;
                r1 += vl;
                r2 += vl;
                r3 += vl;
                outptr0 += vl;
                outptr1 += vl;
                j -= vl;
            }
            r0 += 2 + w;
            r1 += 2 + w;
            r2 += 2 + w;
            r3 += 2 + w;

            outptr0 += outw;
            outptr1 += outw;
        }
        for (; i < outh; i++)
        {
            int j = outw;
            while (j > 0)
            {
                vl = vsetvl_e16m1(j);
                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r20 = vle16_v_f16m1(r2, vl);

                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + 1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + 1, vl);
                vfloat16m1_t _r21 = vle16_v_f16m1(r2 + 1, vl);

                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + 2, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + 2, vl);
                vfloat16m1_t _r22 = vle16_v_f16m1(r2 + 2, vl);

                vfloat16m1_t _sum0 = _bias0;

                _sum0 = vfmacc_vf_f16m1(_sum0, _k0, _r00, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k1, _r01, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k2, _r02, vl);

                _sum0 = vfmacc_vf_f16m1(_sum0, _k3, _r10, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k4, _r11, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k5, _r12, vl);

                _sum0 = vfmacc_vf_f16m1(_sum0, _k6, _r20, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k7, _r21, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k8, _r22, vl);

                vse16_v_f16m1(outptr0, _sum0, vl);

                r0 += vl;
                r1 += vl;
                r2 += vl;
                outptr0 += vl;
                j -= vl;
            }

            r0 += 2;
            r1 += 2;
            r2 += 2;
        }
    }
}

static void convdw3x3s2_fp16sa_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    size_t vl;

    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const int tailstep = w - 2 * outw + w;

    const __fp16* kernel = _kernel;
    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        const __fp16 bias0 = bias ? bias[g] : 0.f;

        const __fp16* kernel0 = kernel + g * 9;

        __fp16* outptr = out;

        const __fp16* img0 = bottom_blob.channel(g);

        const __fp16* r0 = img0;
        const __fp16* r1 = img0 + w;
        const __fp16* r2 = img0 + w * 2;

        __fp16 _k0 = kernel0[0];
        __fp16 _k1 = kernel0[1];
        __fp16 _k2 = kernel0[2];
        __fp16 _k3 = kernel0[3];
        __fp16 _k4 = kernel0[4];
        __fp16 _k5 = kernel0[5];
        __fp16 _k6 = kernel0[6];
        __fp16 _k7 = kernel0[7];
        __fp16 _k8 = kernel0[8];

        vl = vsetvl_e16m1(packn);
        vfloat16m1_t _bias0 = vfmv_v_f_f16m1(bias0, vl);

        int i = 0;
        for (; i < outh; i++)
        {
            int j = outw;
            while (j > 0)
            {
                vl = vsetvl_e16m1(j);
                vfloat16m1_t _r00, _r01, _r02;
                vfloat16m1_t _r10, _r11, _r12;
                vfloat16m1_t _r20, _r21, _r22;
                vlseg2e16_v_f16m1(&_r00, &_r01, r0, vl);
                vlseg2e16_v_f16m1(&_r10, &_r11, r1, vl);
                vlseg2e16_v_f16m1(&_r20, &_r21, r2, vl);
                _r02 = vlse16_v_f16m1(r0 + 2, 2 * sizeof(__fp16), vl);
                _r12 = vlse16_v_f16m1(r1 + 2, 2 * sizeof(__fp16), vl);
                _r22 = vlse16_v_f16m1(r2 + 2, 2 * sizeof(__fp16), vl);

                vfloat16m1_t _sum0 = _bias0;

                _sum0 = vfmacc_vf_f16m1(_sum0, _k0, _r00, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k1, _r01, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k2, _r02, vl);

                _sum0 = vfmacc_vf_f16m1(_sum0, _k3, _r10, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k4, _r11, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k5, _r12, vl);

                _sum0 = vfmacc_vf_f16m1(_sum0, _k6, _r20, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k7, _r21, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, _k8, _r22, vl);

                vse16_v_f16m1(outptr, _sum0, vl);

                r0 += vl * 2;
                r1 += vl * 2;
                r2 += vl * 2;
                outptr += vl;
                j -= vl;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}
