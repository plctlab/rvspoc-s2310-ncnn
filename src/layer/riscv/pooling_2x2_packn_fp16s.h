// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

static void pooling2x2s2_max_packn_fp16s_rvv(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = (w - 2 * outw + w) * packn;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
    {
        const Mat img0 = bottom_blob.channel(q);
        __fp16* outptr = top_blob.channel(q);

        const __fp16* r0 = img0.row<const __fp16>(0);
        const __fp16* r1 = img0.row<const __fp16>(1);

        for (int i = 0; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                vfloat16m1_t _r00 = vle16_v_f16m1(r0 + packn * 0, vl);
                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + packn * 1, vl);
                vfloat16m1_t _r02 = vle16_v_f16m1(r0 + packn * 2, vl);
                vfloat16m1_t _r03 = vle16_v_f16m1(r0 + packn * 3, vl);
                vfloat16m1_t _r04 = vle16_v_f16m1(r0 + packn * 4, vl);
                vfloat16m1_t _r05 = vle16_v_f16m1(r0 + packn * 5, vl);
                vfloat16m1_t _r06 = vle16_v_f16m1(r0 + packn * 6, vl);
                vfloat16m1_t _r07 = vle16_v_f16m1(r0 + packn * 7, vl);
                vfloat16m1_t _r10 = vle16_v_f16m1(r1 + packn * 0, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + packn * 1, vl);
                vfloat16m1_t _r12 = vle16_v_f16m1(r1 + packn * 2, vl);
                vfloat16m1_t _r13 = vle16_v_f16m1(r1 + packn * 3, vl);
                vfloat16m1_t _r14 = vle16_v_f16m1(r1 + packn * 4, vl);
                vfloat16m1_t _r15 = vle16_v_f16m1(r1 + packn * 5, vl);
                vfloat16m1_t _r16 = vle16_v_f16m1(r1 + packn * 6, vl);
                vfloat16m1_t _r17 = vle16_v_f16m1(r1 + packn * 7, vl);

                vfloat16m1_t _max00 = vfmax_vv_f16m1(_r00, _r01, vl);
                vfloat16m1_t _max10 = vfmax_vv_f16m1(_r10, _r11, vl);
                vfloat16m1_t _max01 = vfmax_vv_f16m1(_r02, _r03, vl);
                vfloat16m1_t _max11 = vfmax_vv_f16m1(_r12, _r13, vl);
                vfloat16m1_t _max02 = vfmax_vv_f16m1(_r04, _r05, vl);
                vfloat16m1_t _max12 = vfmax_vv_f16m1(_r14, _r15, vl);
                vfloat16m1_t _max03 = vfmax_vv_f16m1(_r06, _r07, vl);
                vfloat16m1_t _max13 = vfmax_vv_f16m1(_r16, _r17, vl);

                vfloat16m1_t _max0 = vfmax_vv_f16m1(_max00, _max10, vl);
                vfloat16m1_t _max1 = vfmax_vv_f16m1(_max01, _max11, vl);
                vfloat16m1_t _max2 = vfmax_vv_f16m1(_max02, _max12, vl);
                vfloat16m1_t _max3 = vfmax_vv_f16m1(_max03, _max13, vl);

                vse16_v_f16m1(outptr + packn * 0, _max0, vl);
                vse16_v_f16m1(outptr + packn * 1, _max1, vl);
                vse16_v_f16m1(outptr + packn * 2, _max2, vl);
                vse16_v_f16m1(outptr + packn * 3, _max3, vl);

                r0 += packn * (2 * 4);
                r1 += packn * (2 * 4);
                outptr += packn * 4;
            }
            for (; j < outw; j++)
            {
                vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                vfloat16m1_t _r01 = vle16_v_f16m1(r0 + packn, vl);
                vfloat16m1_t _r10 = vle16_v_f16m1(r1, vl);
                vfloat16m1_t _r11 = vle16_v_f16m1(r1 + packn, vl);

                vfloat16m1_t _max0 = vfmax_vv_f16m1(_r00, _r01, vl);
                vfloat16m1_t _max1 = vfmax_vv_f16m1(_r10, _r11, vl);

                vfloat16m1_t _max = vfmax_vv_f16m1(_max0, _max1, vl);

                vse16_v_f16m1(outptr, _max, vl);

                r0 += packn * 2;
                r1 += packn * 2;
                outptr += packn;
            }

            r0 += tailstep;
            r1 += tailstep;
        }
    }
}
