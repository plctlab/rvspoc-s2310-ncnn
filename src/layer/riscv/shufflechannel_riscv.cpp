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

#include "shufflechannel_riscv.h"

#include "layer_type.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include <riscv_usability.h>

#include "cpu.h"

namespace ncnn {

ShuffleChannel_riscv::ShuffleChannel_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif // __riscv_vector
}

int ShuffleChannel_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();
    int elempack = bottom_blob.elempack;

#if __riscv_zfh
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);
#endif

    Option opt_pack = opt;
    opt_pack.blob_allocator = opt.workspace_allocator;

    Mat bottom_blob_unpacked;
    convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);

    Mat top_blob_unpacked;
    int ret = ShuffleChannel::forward(bottom_blob_unpacked, top_blob_unpacked, opt_pack);
    if (ret != 0)
        return ret;

    convert_packing(top_blob_unpacked, top_blob, elempack, opt);
    return 0;
}

int ShuffleChannel_riscv::forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    const int packn = csrr_vlenb() / 2;

    int _group = reverse ? channels * elempack / group : group;

    if (_group == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }
#if __riscv_vector
    if (elempack == packn)
    {
        if (_group == 2 && channels % _group != 0)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int size = w * h;
            size_t elemsize = bottom_blob.elemsize;

            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            int channels_per_group = channels / _group;

            // TODO unroll me
            for (int q = 0; q < channels_per_group; q++)
            {
                const unsigned short* ptr0 = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + q);
                const unsigned short* ptr2 = bottom_blob.channel(channels_per_group + q + 1);
                unsigned short* outptr0 = top_blob.channel(q * 2);
                unsigned short* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    const int vl = vsetvl_e16m1(packn);
                    vuint16m1_t _p0 = vle16_v_u16m1(ptr0, vl);
                    vuint16m1_t _p1 = vle16_v_u16m1(ptr1, vl);
                    vuint16m1_t _p2 = vle16_v_u16m1(ptr2, vl);

                    // index_c906[0] = 0;
                    // index_c906[1] = 12;
                    // index_c906[2] = 1;
                    // index_c906[3] = 13;
                    // index_c906[4] = 2;
                    // index_c906[5] = 14;
                    // index_c906[6] = 3;
                    // index_c906[7] = 15;

                    // index_c906[8] = 4;
                    // index_c906[9] = 16;
                    // index_c906[10] = 5;
                    // index_c906[11] = 17;
                    // index_c906[12] = 6;
                    // index_c906[13] = 18;
                    // index_c906[14] = 7;
                    // index_c906[15] = 19;

                    unsigned short index[16] = {0, 12, 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19};

                    vuint16m4_t _p012 = vundefined_u16m4();
                    _p012 = vset_v_u16m1_u16m4(_p012, 0, _p0);
                    _p012 = vset_v_u16m1_u16m4(_p012, 1, _p1);
                    _p012 = vset_v_u16m1_u16m4(_p012, 2, _p2);

                    // vl = vsetvl_e16m3(3 * packn);
                    vuint16m4_t _idx = vle16_v_u16m4(index, vl * 2);

                    vuint16m4_t _p01 = vrgather_vv_u16m4(_p012, _idx, vl * 2);

                    // vl = vsetvl_e16m1(packn);
                    vse16_v_u16m1(outptr0, vget_v_u16m4_u16m1(_p01, 0), vl);
                    vse16_v_u16m1(outptr1, vget_v_u16m4_u16m1(_p01, 1), vl);

                    // uint16x8_t _p0 = vld1q_u16(ptr0);
                    // uint16x8_t _p1 = vld1q_u16(ptr1);
                    // uint16x8_t _p2 = vld1q_u16(ptr2);

                    // uint16x8_t _p12 = vextq_u16(_p1, _p2, 4);

                    // uint16x8x2_t _p01 = vzipq_u16(_p0, _p12);

                    // vst1q_u16(outptr0, _p01.val[0]);
                    // vst1q_u16(outptr1, _p01.val[1]);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                }
            }

            // handle the last channel
            {
                const unsigned short* ptr0 = bottom_blob.channel(channels_per_group);
                const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + channels_per_group);
                unsigned short* outptr0 = top_blob.channel(channels_per_group * 2);

                ptr1 += 4;

                for (int i = 0; i < size; i++)
                {
                    uint16_t index[8] = {0, 8, 1, 9, 2, 10, 3, 11};
                    int vl = packn / 2;
                    vuint16m1_t _p0 = vle16_v_u16m1(ptr0, vl);
                    vuint16m1_t _p1 = vle16_v_u16m1(ptr1, vl);

                    vuint16m2_t _p01 = vundefined_u16m2();
                    _p01 = vset_v_u16m1_u16m2(_p01, 0, _p0);
                    _p01 = vset_v_u16m1_u16m2(_p01, 1, _p1);

                    vl = 8;
                    vuint16m2_t _idx = vle16_v_u16m2(index, vl);
                    _p01 = vrgather_vv_u16m2(_p01, _idx, vl);
                    vse16_v_u16m2(outptr0, _p01, vl);

                    // uint16x4_t _p0 = vld1_u16(ptr0);
                    // uint16x4_t _p1 = vld1_u16(ptr1);

                    // uint16x4x2_t _p01 = vzip_u16(_p0, _p1);

                    // vst1_u16(outptr0, _p01.val[0]);
                    // vst1_u16(outptr0 + 4, _p01.val[1]);

                    ptr0 += 8;
                    ptr1 += 8;
                    outptr0 += 8;
                }
            }

            return 0;
        }

        if (_group > 4 || channels % _group != 0)
        {
            // slow path for too large group or shuffle inside elempack
            Option opt_pack = opt;
            opt_pack.blob_allocator = opt.workspace_allocator;

            Mat bottom_blob_unpacked;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);

            Mat top_blob_unpacked;
            int ret = ShuffleChannel::forward(bottom_blob_unpacked, top_blob_unpacked, opt_pack);
            if (ret != 0)
                return ret;

            convert_packing(top_blob_unpacked, top_blob, elempack, opt);

            return 0;
        }

        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int size = w * h;
        size_t elemsize = bottom_blob.elemsize;

        top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int channels_per_group = channels / _group;

        if (_group == 2)
        {
            for (int q = 0; q < channels_per_group; q++)
            {
                const unsigned short* ptr0 = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + q);
                unsigned short* outptr0 = top_blob.channel(q * 2);
                unsigned short* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    const size_t vl = packn;
                    vuint16m1_t _p0 = vle16_v_u16m1(ptr0, vl);
                    vuint16m1_t _p1 = vle16_v_u16m1(ptr1, vl);

                    uint16_t index[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};

                    vuint16m2_t _p01 = vundefined_u16m2();
                    _p01 = vset_v_u16m1_u16m2(_p01, 0, _p0);
                    _p01 = vset_v_u16m1_u16m2(_p01, 1, _p1);

                    vuint16m2_t _idx = vle16_v_u16m2(index, vl * 2);
                    _p01 = vrgather_vv_u16m2(_p01, _idx, vl * 2);

                    vse16_v_u16m1(outptr0, vget_v_u16m2_u16m1(_p01, 0), vl);
                    vse16_v_u16m1(outptr1, vget_v_u16m2_u16m1(_p01, 1), vl);

                    // uint16x8_t _p0 = vld1q_u16(ptr0);
                    // uint16x8_t _p1 = vld1q_u16(ptr1);

                    // uint16x8x2_t _p01 = vzipq_u16(_p0, _p1);

                    // vst1q_u16(outptr0, _p01.val[0]);
                    // vst1q_u16(outptr1, _p01.val[1]);

                    ptr0 += 8;
                    ptr1 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                }
            }
        }

        if (_group == 3)
        {
            for (int q = 0; q < channels_per_group; q++)
            {
                const unsigned short* ptr0 = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + q);
                const unsigned short* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                unsigned short* outptr0 = top_blob.channel(q * 3);
                unsigned short* outptr1 = top_blob.channel(q * 3 + 1);
                unsigned short* outptr2 = top_blob.channel(q * 3 + 2);

                for (int i = 0; i < size; i++)
                {
                    const int vl = packn;
                    vuint16m1_t _p0 = vle16_v_u16m1(ptr0, vl);
                    vuint16m1_t _p1 = vle16_v_u16m1(ptr1, vl);
                    vuint16m1_t _p2 = vle16_v_u16m1(ptr2, vl);

                    uint16_t index[24] = {0, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19, 4, 12, 20, 5, 13, 21, 6, 14, 22, 7, 15, 23};

                    vuint16m4_t _p012 = vundefined_u16m4();
                    _p012 = vset_v_u16m1_u16m4(_p012, 0, _p0);
                    _p012 = vset_v_u16m1_u16m4(_p012, 1, _p1);
                    _p012 = vset_v_u16m1_u16m4(_p012, 2, _p2);

                    vuint16m4_t _idx = vle16_v_u16m4(index, vl * 3);

                    _p012 = vrgather_vv_u16m4(_p012, _idx, vl * 3);
                    vse16_v_u16m1(outptr0, vget_v_u16m4_u16m1(_p012, 0), vl);
                    vse16_v_u16m1(outptr1, vget_v_u16m4_u16m1(_p012, 1), vl);
                    vse16_v_u16m1(outptr2, vget_v_u16m4_u16m1(_p012, 2), vl);

                    // uint16x8_t _p0 = vld1q_u16(ptr0);
                    // uint16x8_t _p1 = vld1q_u16(ptr1);
                    // uint16x8_t _p2 = vld1q_u16(ptr2);

                    // TODO figure out a faster way

                    // 01234567        08g19h2a
                    // 89abcdef   ->   i3bj4ck5
                    // ghijklmn        dl6em7fn

                    // uint16x8x3_t _p012;
                    // _p012.val[0] = _p0;
                    // _p012.val[1] = _p1;
                    // _p012.val[2] = _p2;

                    // unsigned short tmp[24];
                    // vst3q_u16(&tmp[0], _p012);

                    // _p0 = vld1q_u16(&tmp[0]);
                    // _p1 = vld1q_u16(&tmp[8]);
                    // _p2 = vld1q_u16(&tmp[16]);

                    // vst1q_u16(outptr0, _p0);
                    // vst1q_u16(outptr1, _p1);
                    // vst1q_u16(outptr2, _p2);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                }
            }
        }

        if (_group == 4)
        {
            for (int q = 0; q < channels_per_group; q++)
            {
                const unsigned short* ptr0 = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + q);
                const unsigned short* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                const unsigned short* ptr3 = bottom_blob.channel(channels_per_group * 3 + q);
                unsigned short* outptr0 = top_blob.channel(q * 4);
                unsigned short* outptr1 = top_blob.channel(q * 4 + 1);
                unsigned short* outptr2 = top_blob.channel(q * 4 + 2);
                unsigned short* outptr3 = top_blob.channel(q * 4 + 3);

                for (int i = 0; i < size; i++)
                {
                    const int vl = packn;

                    vuint16m1_t _p0 = vle16_v_u16m1(ptr0, vl);
                    vuint16m1_t _p1 = vle16_v_u16m1(ptr1, vl);
                    vuint16m1_t _p2 = vle16_v_u16m1(ptr2, vl);
                    vuint16m1_t _p3 = vle16_v_u16m1(ptr3, vl);

                    uint16_t index[32] = {0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27, 4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31};

                    vuint16m4_t _p0123 = vundefined_u16m4();
                    _p0123 = vset_v_u16m1_u16m4(_p0123, 0, _p0);
                    _p0123 = vset_v_u16m1_u16m4(_p0123, 1, _p1);
                    _p0123 = vset_v_u16m1_u16m4(_p0123, 2, _p2);
                    _p0123 = vset_v_u16m1_u16m4(_p0123, 3, _p3);

                    vuint16m4_t _idx = vle16_v_u16m4(index, vl * 4);
                    _p0123 = vrgather_vv_u16m4(_p0123, _idx, vl * 4);
                    vse16_v_u16m1(outptr0, vget_v_u16m4_u16m1(_p0123, 0), vl);
                    vse16_v_u16m1(outptr1, vget_v_u16m4_u16m1(_p0123, 1), vl);
                    vse16_v_u16m1(outptr2, vget_v_u16m4_u16m1(_p0123, 2), vl);
                    vse16_v_u16m1(outptr3, vget_v_u16m4_u16m1(_p0123, 3), vl);
                    // uint16x8_t _p0 = vld1q_u16(ptr0);
                    // uint16x8_t _p1 = vld1q_u16(ptr1);
                    // uint16x8_t _p2 = vld1q_u16(ptr2);
                    // uint16x8_t _p3 = vld1q_u16(ptr3);

                    // transpose 4x4
                    // uint16x8x2_t _p01 = vtrnq_u16(_p0, _p1);
                    // uint16x8x2_t _p23 = vtrnq_u16(_p2, _p3);
                    // uint32x4x2_t _p02 = vtrnq_u32(vreinterpretq_u32_u16(_p01.val[0]), vreinterpretq_u32_u16(_p23.val[0]));
                    // uint32x4x2_t _p13 = vtrnq_u32(vreinterpretq_u32_u16(_p01.val[1]), vreinterpretq_u32_u16(_p23.val[1]));
                    // _p0 = vreinterpretq_u16_u32(_p02.val[0]);
                    // _p1 = vreinterpretq_u16_u32(_p13.val[0]);
                    // _p2 = vreinterpretq_u16_u32(_p02.val[1]);
                    // _p3 = vreinterpretq_u16_u32(_p13.val[1]);

                    // vst1q_u16(outptr0, vcombine_u16(vget_low_u16(_p0), vget_low_u16(_p1)));
                    // vst1q_u16(outptr1, vcombine_u16(vget_low_u16(_p2), vget_low_u16(_p3)));
                    // vst1q_u16(outptr2, vcombine_u16(vget_high_u16(_p0), vget_high_u16(_p1)));
                    // vst1q_u16(outptr3, vcombine_u16(vget_high_u16(_p2), vget_high_u16(_p3)));

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                    outptr3 += 8;
                }
            }
        }

        return 0;
    }
#endif // __riscv_vector

#if __riscv_vector
    if (elempack == 4)
    {
        Option opt_pack = opt;
        opt_pack.blob_allocator = opt.workspace_allocator;

        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);

        Mat top_blob_unpacked;
        int ret = ShuffleChannel::forward(bottom_blob_unpacked, top_blob_unpacked, opt_pack);
        if (ret != 0)
            return ret;

        convert_packing(top_blob_unpacked, top_blob, elempack, opt);

        return 0;
    }
#endif // __riscv_vector

    return ShuffleChannel::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
