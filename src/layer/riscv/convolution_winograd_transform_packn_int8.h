// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv3x3s1_winograd43_transform_input_packn_int8_rvv(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int inch = bottom_blob.c;

    const int w_tiles = (w - 2) / 4;
    const int h_tiles = (h - 2) / 4;
    const int tiles = w_tiles * h_tiles;

    // const int16_t itm[4][4] = {
    //     {4,  0, -5,  0, 1, 0},
    //     {0, -4, -4,  1, 1, 0},
    //     {0,  4, -4, -1, 1, 0},
    //     {0, -2, -1,  2, 1, 0},
    //     {0,  2, -1, -2, 1, 0},
    //     {0,  4,  0, -5, 0, 1}
    // };

    // 0 =  4 * r00 - 5 * r02 + r04
    // 1 =  (-4 * r01 + r03) + (r04 - 4 * r02)
    // 2 = -(-4 * r01 + r03) + (r04 - 4 * r02)
    // 3 = -(2 * r01 - 2 * r03) + (r04 - r02)
    // 4 =  (2 * r01 - 2 * r03) + (r04 - r02)
    // 5 =  4 * r01 - 5 * r03 + r05

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
    {
        const Mat img0 = bottom_blob.channel(q);
        Mat img0_tm = bottom_blob_tm.channel(q);

        // NOTE c99 variable length array
        int16_t tmp[6][6][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const int8_t* r0 = img0.row<const int8_t>(i * 4) + (j * 4) * packn;

                for (int m = 0; m < 6; m++)
                {
                    vint16m2_t _r00_01 = vwcvt_x_x_v_i16m2(vle8_v_i8m1(r0 + packn * 0, vl * 2), vl * 2);
                    vint16m2_t _r02_03 = vwcvt_x_x_v_i16m2(vle8_v_i8m1(r0 + packn * 2, vl * 2), vl * 2);
                    vint16m2_t _r04_05 = vwcvt_x_x_v_i16m2(vle8_v_i8m1(r0 + packn * 4, vl * 2), vl * 2);

                    vint16m1_t _r00 = vget_v_i16m2_i16m1(_r00_01, 0);
                    vint16m1_t _r01 = vget_v_i16m2_i16m1(_r00_01, 1);
                    vint16m1_t _r02 = vget_v_i16m2_i16m1(_r02_03, 0);
                    vint16m1_t _r03 = vget_v_i16m2_i16m1(_r02_03, 1);
                    vint16m1_t _r04 = vget_v_i16m2_i16m1(_r04_05, 0);
                    vint16m1_t _r05 = vget_v_i16m2_i16m1(_r04_05, 1);

                    vint16m1_t _tmp01a = vnmsub_vx_i16m1(_r01, 4, _r03, vl);
                    vint16m1_t _tmp01b = vnmsub_vx_i16m1(_r02, 4, _r04, vl);
                    vint16m1_t _tmp23a = vsll_vx_i16m1(vsub_vv_i16m1(_r01, _r03, vl), 1, vl);
                    vint16m1_t _tmp23b = vsub_vv_i16m1(_r04, _r02, vl);

                    vint16m1_t _tmp0m = vmacc_vx_i16m1(vnmsac_vx_i16m1(_r04, 5, _r02, vl), 4, _r00, vl);
                    vint16m1_t _tmp1m = vadd_vv_i16m1(_tmp01b, _tmp01a, vl);
                    vint16m1_t _tmp2m = vsub_vv_i16m1(_tmp01b, _tmp01a, vl);
                    vint16m1_t _tmp3m = vsub_vv_i16m1(_tmp23b, _tmp23a, vl);
                    vint16m1_t _tmp4m = vadd_vv_i16m1(_tmp23b, _tmp23a, vl);
                    vint16m1_t _tmp5m = vmacc_vx_i16m1(vnmsac_vx_i16m1(_r05, 5, _r03, vl), 4, _r01, vl);

                    vse16_v_i16m1(tmp[0][m], _tmp0m, vl);
                    vse16_v_i16m1(tmp[1][m], _tmp1m, vl);
                    vse16_v_i16m1(tmp[2][m], _tmp2m, vl);
                    vse16_v_i16m1(tmp[3][m], _tmp3m, vl);
                    vse16_v_i16m1(tmp[4][m], _tmp4m, vl);
                    vse16_v_i16m1(tmp[5][m], _tmp5m, vl);

                    r0 += w * packn;
                }

                int16_t* r0_tm_0 = (int16_t*)img0_tm + (i * w_tiles + j) * packn;
                int16_t* r0_tm_1 = r0_tm_0 + tiles * packn;
                int16_t* r0_tm_2 = r0_tm_0 + tiles * packn * 2;
                int16_t* r0_tm_3 = r0_tm_0 + tiles * packn * 3;
                int16_t* r0_tm_4 = r0_tm_0 + tiles * packn * 4;
                int16_t* r0_tm_5 = r0_tm_0 + tiles * packn * 5;

                for (int m = 0; m < 6; m++)
                {
                    vint16m1_t _r00 = vle16_v_i16m1(tmp[m][0], vl);
                    vint16m1_t _r01 = vle16_v_i16m1(tmp[m][1], vl);
                    vint16m1_t _r02 = vle16_v_i16m1(tmp[m][2], vl);
                    vint16m1_t _r03 = vle16_v_i16m1(tmp[m][3], vl);
                    vint16m1_t _r04 = vle16_v_i16m1(tmp[m][4], vl);
                    vint16m1_t _r05 = vle16_v_i16m1(tmp[m][5], vl);

                    vint16m1_t _tmp01a = vnmsub_vx_i16m1(_r01, 4, _r03, vl);
                    vint16m1_t _tmp01b = vnmsub_vx_i16m1(_r02, 4, _r04, vl);
                    vint16m1_t _tmp23a = vsll_vx_i16m1(vsub_vv_i16m1(_r01, _r03, vl), 1, vl);
                    vint16m1_t _tmp23b = vsub_vv_i16m1(_r04, _r02, vl);

                    vint16m1_t _tmp0m = vmacc_vx_i16m1(vnmsac_vx_i16m1(_r04, 5, _r02, vl), 4, _r00, vl);
                    vint16m1_t _tmp1m = vadd_vv_i16m1(_tmp01b, _tmp01a, vl);
                    vint16m1_t _tmp2m = vsub_vv_i16m1(_tmp01b, _tmp01a, vl);
                    vint16m1_t _tmp3m = vsub_vv_i16m1(_tmp23b, _tmp23a, vl);
                    vint16m1_t _tmp4m = vadd_vv_i16m1(_tmp23b, _tmp23a, vl);
                    vint16m1_t _tmp5m = vmacc_vx_i16m1(vnmsac_vx_i16m1(_r05, 5, _r03, vl), 4, _r01, vl);

                    vse16_v_i16m1(r0_tm_0, _tmp0m, vl);
                    vse16_v_i16m1(r0_tm_1, _tmp1m, vl);
                    vse16_v_i16m1(r0_tm_2, _tmp2m, vl);
                    vse16_v_i16m1(r0_tm_3, _tmp3m, vl);
                    vse16_v_i16m1(r0_tm_4, _tmp4m, vl);
                    vse16_v_i16m1(r0_tm_5, _tmp5m, vl);

                    r0_tm_0 += tiles * packn * 6;
                    r0_tm_1 += tiles * packn * 6;
                    r0_tm_2 += tiles * packn * 6;
                    r0_tm_3 += tiles * packn * 6;
                    r0_tm_4 += tiles * packn * 6;
                    r0_tm_5 += tiles * packn * 6;
                }
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_output_packn_int8_rvv(const Mat& top_blob_tm, Mat& top_blob, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e32m2(packn);

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int w_tiles = outw / 4;
    const int h_tiles = outh / 4;
    const int tiles = w_tiles * h_tiles;

    // const int otm[4][6] = {
    //     {1, 1,  1, 1,  1, 0},
    //     {0, 1, -1, 2, -2, 0},
    //     {0, 1,  1, 4,  4, 0},
    //     {0, 1, -1, 8, -8, 1}
    // };

    // 0 = r00 + (r01 + r02) + (r03 + r04)
    // 1 =       (r01 - r02) + (r03 - r04) * 2
    // 2 =       (r01 + r02) + (r03 + r04) * 4
    // 3 = r05 + (r01 - r02) + (r03 - r04) * 8

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        const Mat out0_tm = top_blob_tm.channel(p);
        Mat out0 = top_blob.channel(p);

        // NOTE variable length array
        int32_t tmp[4][6][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const int32_t* output0_tm_0 = (const int32_t*)out0_tm + (i * w_tiles + j) * packn;
                const int32_t* output0_tm_1 = output0_tm_0 + tiles * packn;
                const int32_t* output0_tm_2 = output0_tm_0 + tiles * packn * 2;
                const int32_t* output0_tm_3 = output0_tm_0 + tiles * packn * 3;
                const int32_t* output0_tm_4 = output0_tm_0 + tiles * packn * 4;
                const int32_t* output0_tm_5 = output0_tm_0 + tiles * packn * 5;

                int32_t* output0 = out0.row<int32_t>(i * 4) + (j * 4) * packn;

                for (int m = 0; m < 5; m++)
                {
                    vint32m2_t _r00 = vle32_v_i32m2(output0_tm_0, vl);
                    vint32m2_t _r01 = vle32_v_i32m2(output0_tm_1, vl);
                    vint32m2_t _r02 = vle32_v_i32m2(output0_tm_2, vl);
                    vint32m2_t _r03 = vle32_v_i32m2(output0_tm_3, vl);
                    vint32m2_t _r04 = vle32_v_i32m2(output0_tm_4, vl);
                    vint32m2_t _r05 = vle32_v_i32m2(output0_tm_5, vl);

                    vint32m2_t _tmp02a = vadd_vv_i32m2(_r01, _r02, vl);
                    vint32m2_t _tmp02b = vadd_vv_i32m2(_r03, _r04, vl);
                    vint32m2_t _tmp13a = vsub_vv_i32m2(_r01, _r02, vl);
                    vint32m2_t _tmp13b = vsub_vv_i32m2(_r03, _r04, vl);

                    vint32m2_t _tmp0m = vadd_vv_i32m2(vadd_vv_i32m2(_r00, _tmp02a, vl), _tmp02b, vl);
                    vint32m2_t _tmp1m = vmadd_vx_i32m2(_tmp13b, 2, _tmp13a, vl);
                    vint32m2_t _tmp2m = vmadd_vx_i32m2(_tmp02b, 4, _tmp02a, vl);
                    vint32m2_t _tmp3m = vmadd_vx_i32m2(_tmp13b, 8, vmadd_vx_i32m2(_r05, 4, _tmp13a, vl), vl);

                    vse32_v_i32m2(tmp[0][m], _tmp0m, vl);
                    vse32_v_i32m2(tmp[1][m], _tmp1m, vl);
                    vse32_v_i32m2(tmp[2][m], _tmp2m, vl);
                    vse32_v_i32m2(tmp[3][m], _tmp3m, vl);

                    output0_tm_0 += tiles * packn * 6;
                    output0_tm_1 += tiles * packn * 6;
                    output0_tm_2 += tiles * packn * 6;
                    output0_tm_3 += tiles * packn * 6;
                    output0_tm_4 += tiles * packn * 6;
                    output0_tm_5 += tiles * packn * 6;
                }
                for (int m = 5; m < 6; m++)
                {
                    vint32m2_t _r00 = vle32_v_i32m2(output0_tm_0, vl);
                    vint32m2_t _r01 = vle32_v_i32m2(output0_tm_1, vl);
                    vint32m2_t _r02 = vle32_v_i32m2(output0_tm_2, vl);
                    vint32m2_t _r03 = vle32_v_i32m2(output0_tm_3, vl);
                    vint32m2_t _r04 = vle32_v_i32m2(output0_tm_4, vl);
                    vint32m2_t _r05 = vle32_v_i32m2(output0_tm_5, vl);

                    vint32m2_t _tmp02a = vadd_vv_i32m2(_r01, _r02, vl);
                    vint32m2_t _tmp02b = vadd_vv_i32m2(_r03, _r04, vl);
                    vint32m2_t _tmp13a = vsub_vv_i32m2(_r01, _r02, vl);
                    vint32m2_t _tmp13b = vsub_vv_i32m2(_r03, _r04, vl);

                    vint32m2_t _tmp0m = vadd_vv_i32m2(vadd_vv_i32m2(_r00, _tmp02a, vl), _tmp02b, vl);
                    vint32m2_t _tmp1m = vmadd_vx_i32m2(_tmp13b, 2, _tmp13a, vl);
                    vint32m2_t _tmp2m = vmadd_vx_i32m2(_tmp02b, 4, _tmp02a, vl);
                    vint32m2_t _tmp3m = vmadd_vx_i32m2(_tmp13b, 8, vmadd_vx_i32m2(_r05, 4, _tmp13a, vl), vl);

                    _tmp0m = vsll_vx_i32m2(_tmp0m, 2, vl);
                    _tmp1m = vsll_vx_i32m2(_tmp1m, 2, vl);
                    _tmp2m = vsll_vx_i32m2(_tmp2m, 2, vl);
                    _tmp3m = vsll_vx_i32m2(_tmp3m, 2, vl);

                    vse32_v_i32m2(tmp[0][m], _tmp0m, vl);
                    vse32_v_i32m2(tmp[1][m], _tmp1m, vl);
                    vse32_v_i32m2(tmp[2][m], _tmp2m, vl);
                    vse32_v_i32m2(tmp[3][m], _tmp3m, vl);

                    output0_tm_0 += tiles * packn * 6;
                    output0_tm_1 += tiles * packn * 6;
                    output0_tm_2 += tiles * packn * 6;
                    output0_tm_3 += tiles * packn * 6;
                    output0_tm_4 += tiles * packn * 6;
                    output0_tm_5 += tiles * packn * 6;
                }

                for (int m = 0; m < 4; m++)
                {
                    vint32m2_t _r00 = vle32_v_i32m2(tmp[m][0], vl);
                    vint32m2_t _r01 = vle32_v_i32m2(tmp[m][1], vl);
                    vint32m2_t _r02 = vle32_v_i32m2(tmp[m][2], vl);
                    vint32m2_t _r03 = vle32_v_i32m2(tmp[m][3], vl);
                    vint32m2_t _r04 = vle32_v_i32m2(tmp[m][4], vl);
                    vint32m2_t _r05 = vle32_v_i32m2(tmp[m][5], vl);

                    vint32m2_t _tmp02a = vadd_vv_i32m2(_r01, _r02, vl);
                    vint32m2_t _tmp02b = vadd_vv_i32m2(_r03, _r04, vl);
                    vint32m2_t _tmp13a = vsub_vv_i32m2(_r01, _r02, vl);
                    vint32m2_t _tmp13b = vsub_vv_i32m2(_r03, _r04, vl);

                    vint32m2_t _out00 = vadd_vv_i32m2(vadd_vv_i32m2(_r00, _tmp02a, vl), _tmp02b, vl);
                    vint32m2_t _out01 = vmadd_vx_i32m2(_tmp13b, 2, _tmp13a, vl);
                    vint32m2_t _out02 = vmadd_vx_i32m2(_tmp02b, 4, _tmp02a, vl);
                    vint32m2_t _out03 = vmadd_vx_i32m2(_tmp13b, 8, vadd_vv_i32m2(_tmp13a, _r05, vl), vl);

                    // fastdiv: 2^32 / 7456540 = 576.00003433
                    _out00 = vmulh_vx_i32m2(_out00, 7456540, vl);
                    _out01 = vmulh_vx_i32m2(_out01, 7456540, vl);
                    _out02 = vmulh_vx_i32m2(_out02, 7456540, vl);
                    _out03 = vmulh_vx_i32m2(_out03, 7456540, vl);

                    vse32_v_i32m2(output0 + packn * 0, _out00, vl);
                    vse32_v_i32m2(output0 + packn * 1, _out01, vl);
                    vse32_v_i32m2(output0 + packn * 2, _out02, vl);
                    vse32_v_i32m2(output0 + packn * 3, _out03, vl);

                    output0 += outw * packn;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_input_packn_int8_rvv(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int inch = bottom_blob.c;

    const int w_tiles = (w - 2) / 2;
    const int h_tiles = (h - 2) / 2;
    const int tiles = w_tiles * h_tiles;

    // const float itm[4][4] = {
    //     {1.0f,  0.0f, -1.0f,  0.0f},
    //     {0.0f,  1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  0.00f, 1.0f}
    // };

    // 0 = r00 - r02
    // 1 = r01 + r02
    // 2 = r02 - r01
    // 3 = r03 - r01

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
    {
        const Mat img0 = bottom_blob.channel(q);
        Mat img0_tm = bottom_blob_tm.channel(q);

        // NOTE c99 variable length array
        int16_t tmp[4][4][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const int8_t* r0 = img0.row<const int8_t>(i * 2) + (j * 2) * packn;

                for (int m = 0; m < 4; m++)
                {
                    vint16m1_t _r00 = vget_v_i16m2_i16m1(vwcvt_x_x_v_i16m2(vle8_v_i8m1(r0, vl), vl), 0);
                    vint16m1_t _r01 = vget_v_i16m2_i16m1(vwcvt_x_x_v_i16m2(vle8_v_i8m1(r0 + packn, vl), vl), 0);
                    vint16m1_t _r02 = vget_v_i16m2_i16m1(vwcvt_x_x_v_i16m2(vle8_v_i8m1(r0 + packn * 2, vl), vl), 0);
                    vint16m1_t _r03 = vget_v_i16m2_i16m1(vwcvt_x_x_v_i16m2(vle8_v_i8m1(r0 + packn * 3, vl), vl), 0);
                    // vint16m1_t _r00 = vle16_v_i16m1(r0, vl);
                    // vint16m1_t _r01 = vle16_v_i16m1(r0 + packn, vl);
                    // vint16m1_t _r02 = vle16_v_i16m1(r0 + packn * 2, vl);
                    // vint16m1_t _r03 = vle16_v_i16m1(r0 + packn * 3, vl);

                    vint16m1_t _tmp0m = vsub_vv_i16m1(_r00, _r02, vl);
                    vint16m1_t _tmp1m = vadd_vv_i16m1(_r01, _r02, vl);
                    vint16m1_t _tmp2m = vsub_vv_i16m1(_r02, _r01, vl);
                    vint16m1_t _tmp3m = vsub_vv_i16m1(_r03, _r01, vl);

                    vse16_v_i16m1(tmp[0][m], _tmp0m, vl);
                    vse16_v_i16m1(tmp[1][m], _tmp1m, vl);
                    vse16_v_i16m1(tmp[2][m], _tmp2m, vl);
                    vse16_v_i16m1(tmp[3][m], _tmp3m, vl);

                    r0 += w * packn;
                }

                int16_t* r0_tm_0 = (int16_t*)img0_tm + (i * w_tiles + j) * packn;
                int16_t* r0_tm_1 = r0_tm_0 + tiles * packn;
                int16_t* r0_tm_2 = r0_tm_0 + tiles * packn * 2;
                int16_t* r0_tm_3 = r0_tm_0 + tiles * packn * 3;

                for (int m = 0; m < 4; m++)
                {
                    vint16m1_t _tmp00 = vle16_v_i16m1(tmp[m][0], vl);
                    vint16m1_t _tmp01 = vle16_v_i16m1(tmp[m][1], vl);
                    vint16m1_t _tmp02 = vle16_v_i16m1(tmp[m][2], vl);
                    vint16m1_t _tmp03 = vle16_v_i16m1(tmp[m][3], vl);

                    vint16m1_t _r0tm0 = vsub_vv_i16m1(_tmp00, _tmp02, vl);
                    vint16m1_t _r0tm1 = vadd_vv_i16m1(_tmp01, _tmp02, vl);
                    vint16m1_t _r0tm2 = vsub_vv_i16m1(_tmp02, _tmp01, vl);
                    vint16m1_t _r0tm3 = vsub_vv_i16m1(_tmp03, _tmp01, vl);

                    vse16_v_i16m1(r0_tm_0, _r0tm0, vl);
                    vse16_v_i16m1(r0_tm_1, _r0tm1, vl);
                    vse16_v_i16m1(r0_tm_2, _r0tm2, vl);
                    vse16_v_i16m1(r0_tm_3, _r0tm3, vl);

                    r0_tm_0 += tiles * packn * 4;
                    r0_tm_1 += tiles * packn * 4;
                    r0_tm_2 += tiles * packn * 4;
                    r0_tm_3 += tiles * packn * 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_output_packn_int8_rvv(const Mat& top_blob_tm, Mat& top_blob, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int w_tiles = outw / 2;
    const int h_tiles = outh / 2;
    const int tiles = w_tiles * h_tiles;

    // const float otm[2][4] = {
    //     {1.0f,  1.0f,  1.0f,  0.0f},
    //     {0.0f,  1.0f, -1.0f,  1.0f}
    // };

    // 0 = r00 + r01 + r02
    // 1 = r01 - r02 + r03

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        const Mat out0_tm = top_blob_tm.channel(p);
        Mat out0 = top_blob.channel(p);

        // NOTE variable length array
        int32_t tmp[2][4][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const int32_t* output0_tm_0 = (const int32_t*)out0_tm + (i * w_tiles + j) * packn;
                const int32_t* output0_tm_1 = output0_tm_0 + tiles * packn;
                const int32_t* output0_tm_2 = output0_tm_0 + tiles * packn * 2;
                const int32_t* output0_tm_3 = output0_tm_0 + tiles * packn * 3;

                int32_t* output0 = out0.row<int32_t>(i * 2) + (j * 2) * packn;

                for (int m = 0; m < 4; m++)
                {
                    vint32m2_t _out0tm0 = vle32_v_i32m2(output0_tm_0, vl);
                    vint32m2_t _out0tm1 = vle32_v_i32m2(output0_tm_1, vl);
                    vint32m2_t _out0tm2 = vle32_v_i32m2(output0_tm_2, vl);
                    vint32m2_t _out0tm3 = vle32_v_i32m2(output0_tm_3, vl);

                    vint32m2_t _tmp0m = vadd_vv_i32m2(vadd_vv_i32m2(_out0tm0, _out0tm1, vl), _out0tm2, vl);
                    vint32m2_t _tmp1m = vadd_vv_i32m2(vsub_vv_i32m2(_out0tm1, _out0tm2, vl), _out0tm3, vl);

                    vse32_v_i32m2(tmp[0][m], _tmp0m, vl);
                    vse32_v_i32m2(tmp[1][m], _tmp1m, vl);

                    output0_tm_0 += tiles * packn * 4;
                    output0_tm_1 += tiles * packn * 4;
                    output0_tm_2 += tiles * packn * 4;
                    output0_tm_3 += tiles * packn * 4;
                }

                for (int m = 0; m < 2; m++)
                {
                    vint32m2_t _tmp00 = vle32_v_i32m2(tmp[m][0], vl);
                    vint32m2_t _tmp01 = vle32_v_i32m2(tmp[m][1], vl);
                    vint32m2_t _tmp02 = vle32_v_i32m2(tmp[m][2], vl);
                    vint32m2_t _tmp03 = vle32_v_i32m2(tmp[m][3], vl);

                    vint32m2_t _out00 = vadd_vv_i32m2(vadd_vv_i32m2(_tmp00, _tmp01, vl), _tmp02, vl);
                    vint32m2_t _out01 = vadd_vv_i32m2(vsub_vv_i32m2(_tmp01, _tmp02, vl), _tmp03, vl);

                    _out00 = vsra_vx_i32m2(_out00, 2, vl);
                    _out01 = vsra_vx_i32m2(_out01, 2, vl);

                    vse32_v_i32m2(output0, _out00, vl);
                    vse32_v_i32m2(output0 + packn, _out01, vl);

                    output0 += outw * packn;
                }
            }
        }
    }
}
