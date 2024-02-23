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

#include "convolutiondepthwise_riscv.h"

#include "cpu.h"
#include "layer_type.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

namespace ncnn {

#include "convolutiondepthwise_3x3.h"

#if NCNN_INT8
#include "convolutiondepthwise_3x3_pack8_int8.h"
#endif // NCNN_INT8

#if __riscv_vector
#include "convolutiondepthwise_3x3_packn.h"
#include "convolutiondepthwise_5x5_packn.h"

#if __riscv_zfh
#include "convolutiondepthwise_3x3_fp16s.h"
#include "convolutiondepthwise_3x3_packn_fp16s.h"
#include "convolutiondepthwise_5x5_packn_fp16s.h"
#endif
#endif // __riscv_vector

ConvolutionDepthWise_riscv::ConvolutionDepthWise_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif // __riscv_vector

    activation = 0;
}

int ConvolutionDepthWise_riscv::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        // TODO implement int8
        return create_pipeline_int8(opt);
    }
#endif

#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
#endif

    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        int elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            elempack = channels % packn == 0 ? packn : 1;
        }
#endif

#if __riscv_vector
        // packn
        if (elempack == packn)
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, group);
            convert_packing(weight_data_r2, weight_data_tm, packn, opt);
        }
#endif // __riscv_vector

        if (elempack == 1)
        {
            weight_data_tm = weight_data;
        }

        weight_data.release();

        return 0;
    }

    // group convolution
    create_group_ops(opt);

    weight_data.release();

    return 0;
}

int ConvolutionDepthWise_riscv::create_group_ops(const Option& opt)
{
    // create Convolution op for each group
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    for (int i = 0; i < (int)group_ops.size(); i++)
        delete group_ops[i];

    group_ops.clear();

    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

    group_ops.resize(group);

    for (int g = 0; g < group; g++)
    {
        Mat weight_data_g = weight_data.range(maxk * channels_g * num_output_g * g, maxk * channels_g * num_output_g).clone();
        Mat bias_data_g;
        if (bias_term)
            bias_data_g = bias_data.range(num_output_g * g, num_output_g);

        ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Convolution);

        // set param
        ncnn::ParamDict pd;
        pd.set(0, num_output_g); // num_output
        pd.set(1, kernel_w);
        pd.set(11, kernel_h);
        pd.set(2, dilation_w);
        pd.set(12, dilation_h);
        pd.set(3, stride_w);
        pd.set(13, stride_h);
        pd.set(4, 0);  // pad_w
        pd.set(14, 0); // pad_h
        pd.set(5, bias_term);
        pd.set(6, maxk * channels_g * num_output_g); // weight_data_size
        pd.set(8, int8_scale_term);
        pd.set(9, activation_type);
        pd.set(10, activation_params);

        op->load_param(pd);

        // set weights
        if (bias_term)
        {
            ncnn::Mat weights[5];
            weights[0] = weight_data_g;
            weights[1] = bias_data_g;

#if NCNN_INT8
            if (int8_scale_term)
            {
                Mat weight_data_int8_scales_g(num_output_g);
                weight_data_int8_scales_g.fill(weight_data_int8_scales[g]);
                weights[2] = weight_data_int8_scales_g;
                weights[3] = bottom_blob_int8_scales.range(g, 1);
            }
            if (int8_scale_term > 100)
            {
                weights[4] = top_blob_int8_scales.range(g, 1);
            }
#endif

            op->load_model(ModelBinFromMatArray(weights));
        }
        else
        {
            ncnn::Mat weights[4];
            weights[0] = weight_data_g;

#if NCNN_INT8
            if (int8_scale_term)
            {
                Mat weight_data_int8_scales_g(num_output_g);
                weight_data_int8_scales_g.fill(weight_data_int8_scales[g]);
                weights[1] = weight_data_int8_scales_g;
                weights[2] = bottom_blob_int8_scales.range(g, 1);
            }
            if (int8_scale_term > 100)
            {
                weights[3] = top_blob_int8_scales.range(g, 1);
            }
#endif

            op->load_model(ModelBinFromMatArray(weights));
        }

        op->create_pipeline(opt);

        group_ops[g] = op;
    }

    return 0;
}

int ConvolutionDepthWise_riscv::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

    for (int i = 0; i < (int)group_ops.size(); i++)
    {
        group_ops[i]->destroy_pipeline(opt);
        delete group_ops[i];
    }
    group_ops.clear();

    return 0;
}

int ConvolutionDepthWise_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && int8_scale_term)
    {
        return forward_int8(bottom_blob, top_blob, opt);
    }
#endif

    int elembits = bottom_blob.elembits();

#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = vsetvl_e32m1(packn);
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw5x5s1_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw5x5s2_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else
            {
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
                            space_ofs[p1] = p2;
                            p1++;
                            p2 += dilation_w;
                        }
                        p2 += gap;
                    }
                }

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < channels; g++)
                {
                    float* outptr = top_blob.channel(g);
                    const float* kptr = (const float*)weight_data_tm + maxk * g * packn;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);

                            if (bias_term)
                            {
                                _sum = vle32_v_f32m1((const float*)bias_data + g * packn, vl);
                            }

                            const float* sptr = m.row(i * stride_h) + j * stride_w * packn;

                            for (int k = 0; k < maxk; k++)
                            {
                                vfloat32m1_t _val = vle32_v_f32m1(sptr + space_ofs[k] * packn, vl);
                                vfloat32m1_t _w = vle32_v_f32m1(kptr + k * packn, vl);
                                _sum = vfmacc_vv_f32m1(_sum, _val, _w, vl);
                            }

                            _sum = activation_ps(_sum, activation_type, activation_params, vl);

                            vse32_v_f32m1(outptr + j * packn, _sum, vl);
                        }

                        outptr += outw * packn;
                    }
                }
            }
        }
#endif // __riscv_vector

        if (elempack == 1)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else
            {
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
                            space_ofs[p1] = p2;
                            p1++;
                            p2 += dilation_w;
                        }
                        p2 += gap;
                    }
                }

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < group; g++)
                {
                    float* outptr = top_blob.channel(g);
                    const float* kptr = (const float*)weight_data_tm + maxk * g;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float sum = 0.f;

                            if (bias_term)
                                sum = bias_data[g];

                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = (float)sptr[space_ofs[k]];
                                float w = (float)kptr[k];
                                sum += val * w;
                            }

                            sum = activation_ss(sum, activation_type, activation_params);

                            outptr[j] = sum;
                        }

                        outptr += outw;
                    }
                }
            }
        }

        return 0;
    }

    // group convolution
    const int channels_g = channels * elempack / group;
    const int num_output_g = num_output / group;

    int g_elempack = 1;
    int out_g_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        g_elempack = channels_g % packn == 0 ? packn : 1;
        out_g_elempack = num_output_g % packn == 0 ? packn : 1;
    }
#endif

    // unpacking
    Mat bottom_blob_bordered_unpacked = bottom_blob_bordered;
    if (elempack > g_elempack)
    {
        Option opt_p = opt;
        opt_p.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob_bordered, bottom_blob_bordered_unpacked, 1, opt_p);
    }

    Mat top_blob_unpacked = top_blob;
    if (out_g_elempack < out_elempack)
    {
        top_blob_unpacked.create(outw, outh, num_output, out_elemsize / out_elempack, 1, opt.workspace_allocator);
        if (top_blob_unpacked.empty())
            return -100;
    }

    for (int g = 0; g < group; g++)
    {
        const Mat bottom_blob_bordered_g = bottom_blob_bordered_unpacked.channel_range(channels_g * g / g_elempack, channels_g / g_elempack);
        Mat top_blob_g = top_blob_unpacked.channel_range(num_output_g * g / out_g_elempack, num_output_g / out_g_elempack);

        const ncnn::Layer* op = group_ops[g];

        Option opt_g = opt;
        opt_g.blob_allocator = top_blob_unpacked.allocator;

        // forward
        op->forward(bottom_blob_bordered_g, top_blob_g, opt_g);
    }

    // packing
    if (out_g_elempack < out_elempack)
    {
        convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
    }
    else
    {
        top_blob = top_blob_unpacked;
    }

    return 0;
}

int ConvolutionDepthWise_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _kernel_w = _weight_data.w;
    const int _kernel_h = _weight_data.h;
    const int _num_output = _weight_data.c * _weight_data.elempack;

    Mat weight_data_flattened;
    flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

#if NCNN_RVV
    if (opt.use_fp16_storage && cpu_support_riscv_v() && cpu_support_riscv_zfh() && weight_data_flattened.elembits() == 16)
    {
        Mat weight_data_flattened_fp32;
        cast_float16_to_float32(weight_data_flattened, weight_data_flattened_fp32, opt);
        weight_data_flattened = weight_data_flattened_fp32;
    }
#endif // NCNN_RVV

    // weight_data_flattened as pack1
    weight_data_flattened.w *= weight_data_flattened.elempack;
    weight_data_flattened.elemsize /= weight_data_flattened.elempack;
    weight_data_flattened.elempack = 1;

    Mat bias_data_flattened;
    if (bias_term)
    {
        const Mat& _bias_data = bottom_blobs[2];
        flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty())
            return -100;

#if NCNN_RVV
        if (opt.use_fp16_storage && cpu_support_riscv_v() && cpu_support_riscv_zfh() && bias_data_flattened.elembits() == 16)
        {
            Mat bias_data_flattened_fp32;
            cast_float16_to_float32(bias_data_flattened, bias_data_flattened_fp32, opt);
            bias_data_flattened = bias_data_flattened_fp32;
        }
#endif // NCNN_RVV

        // bias_data_flattened as pack1
        bias_data_flattened.w *= bias_data_flattened.elempack;
        bias_data_flattened.elemsize /= bias_data_flattened.elempack;
        bias_data_flattened.elempack = 1;
    }

    ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::ConvolutionDepthWise);

    ncnn::ParamDict pd;
    pd.set(0, _num_output);
    pd.set(1, _kernel_w);
    pd.set(11, _kernel_h);
    pd.set(2, dilation_w);
    pd.set(12, dilation_h);
    pd.set(3, stride_w);
    pd.set(13, stride_h);
    pd.set(4, pad_left);
    pd.set(15, pad_right);
    pd.set(14, pad_top);
    pd.set(16, pad_bottom);
    pd.set(18, pad_value);
    pd.set(5, bias_term);
    pd.set(6, weight_data_flattened.w);
    pd.set(7, group);
    pd.set(8, int8_scale_term);
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    op->load_param(pd);

    ncnn::Mat weights[2];
    weights[0] = weight_data_flattened;
    weights[1] = bias_data_flattened;

    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    op->forward(bottom_blob, top_blob, opt);

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

#if __riscv_vector && __riscv_zfh
int ConvolutionDepthWise_riscv::create_pipeline_fp16s(const Option& opt)
{
    const int packn = csrr_vlenb() / 2;

    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        int elempack = 1;
        if (opt.use_packing_layout)
        {
            elempack = channels % packn == 0 ? packn : 1;
        }

        // packn
        if (elempack == packn)
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, group);
            Mat weight_data_r2_packed;
            convert_packing(weight_data_r2, weight_data_r2_packed, packn, opt);

            ncnn::cast_float32_to_float16(weight_data_r2_packed, weight_data_tm, opt);
        }

        if (elempack == 1)
        {
            ncnn::cast_float32_to_float16(weight_data, weight_data_tm, opt);
        }

        ncnn::cast_float32_to_float16(bias_data, bias_data_fp16, opt);

        weight_data.release();

        return 0;
    }

    // group convolution
    create_group_ops(opt);

    weight_data.release();

    return 0;
}

int ConvolutionDepthWise_riscv::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = (opt.use_packing_layout && num_output % packn == 0) ? packn : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
        if (elempack == packn)
        {
            {
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
                            space_ofs[p1] = p2;
                            p1++;
                            p2 += dilation_w;
                        }
                        p2 += gap;
                    }
                }

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < channels; g++)
                {
                    __fp16* outptr = top_blob.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g * packn;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            vfloat32m2_t _sum = vfmv_v_f_f32m2(0.f, vl);

                            if (bias_term)
                            {
                                _sum = vle32_v_f32m2((const float*)bias_data + g * packn, vl);
                            }

                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * packn;

                            for (int k = 0; k < maxk; k++)
                            {
                                vfloat16m1_t _val = vle16_v_f16m1(sptr + space_ofs[k] * packn, vl);
                                vfloat16m1_t _w = vle16_v_f16m1(kptr + k * packn, vl);
                                _sum = vfwmacc_vv_f32m2(_sum, _val, _w, vl);
                            }

                            _sum = activation_ps(_sum, activation_type, activation_params, vl);

                            vse16_v_f16m1(outptr + j * packn, vfncvt_f_f_w_f16m1(_sum, vl), vl);
                        }

                        outptr += outw * packn;
                    }
                }
            }
        }

        if (elempack == 1)
        {
            {
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
                            space_ofs[p1] = p2;
                            p1++;
                            p2 += dilation_w;
                        }
                        p2 += gap;
                    }
                }

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < group; g++)
                {
                    __fp16* outptr = top_blob.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float sum = 0.f;

                            if (bias_term)
                                sum = bias_data[g];

                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = (float)sptr[space_ofs[k]];
                                float w = (float)kptr[k];
                                sum += val * w;
                            }

                            sum = activation_ss(sum, activation_type, activation_params);

                            outptr[j] = (__fp16)sum;
                        }

                        outptr += outw;
                    }
                }
            }
        }

        return 0;
    }

    // group convolution
    const int channels_g = channels * elempack / group;
    const int num_output_g = num_output / group;

    int g_elempack = (opt.use_packing_layout && channels_g % packn == 0) ? packn : 1;
    int out_g_elempack = (opt.use_packing_layout && num_output_g % packn == 0) ? packn : 1;

    // unpacking
    Mat bottom_blob_bordered_unpacked = bottom_blob_bordered;
    if (elempack > g_elempack)
    {
        Option opt_p = opt;
        opt_p.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob_bordered, bottom_blob_bordered_unpacked, 1, opt_p);
    }

    Mat top_blob_unpacked = top_blob;
    if (out_g_elempack < out_elempack)
    {
        top_blob_unpacked.create(outw, outh, num_output, out_elemsize / out_elempack, 1, opt.workspace_allocator);
        if (top_blob_unpacked.empty())
            return -100;
    }

    for (int g = 0; g < group; g++)
    {
        const Mat bottom_blob_bordered_g = bottom_blob_bordered_unpacked.channel_range(channels_g * g / g_elempack, channels_g / g_elempack);
        Mat top_blob_g = top_blob_unpacked.channel_range(num_output_g * g / out_g_elempack, num_output_g / out_g_elempack);

        const ncnn::Layer* op = group_ops[g];

        Option opt_g = opt;
        opt_g.blob_allocator = top_blob_unpacked.allocator;

        // forward
        op->forward(bottom_blob_bordered_g, top_blob_g, opt_g);
    }

    // packing
    if (out_g_elempack < out_elempack)
    {
        convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
    }
    else
    {
        top_blob = top_blob_unpacked;
    }

    return 0;
}

int ConvolutionDepthWise_riscv::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = (opt.use_packing_layout && num_output % packn == 0) ? packn : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
        if (elempack == packn)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_packn_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_packn_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw5x5s1_packn_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw5x5s2_packn_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else
            {
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
                            space_ofs[p1] = p2;
                            p1++;
                            p2 += dilation_w;
                        }
                        p2 += gap;
                    }
                }

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < channels; g++)
                {
                    __fp16* outptr = top_blob.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g * packn;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            vfloat16m1_t _sum = vfmv_v_f_f16m1((__fp16)0.f, vl);

                            if (bias_term)
                            {
                                _sum = vle16_v_f16m1((const __fp16*)bias_data_fp16 + g * packn, vl);
                            }

                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * packn;

                            for (int k = 0; k < maxk; k++)
                            {
                                vfloat16m1_t _val = vle16_v_f16m1(sptr + space_ofs[k] * packn, vl);
                                vfloat16m1_t _w = vle16_v_f16m1(kptr + k * packn, vl);
                                _sum = vfmacc_vv_f16m1(_sum, _val, _w, vl);
                            }

                            _sum = activation_ps(_sum, activation_type, activation_params, vl);

                            vse16_v_f16m1(outptr + j * packn, _sum, vl);
                        }

                        outptr += outw * packn;
                    }
                }
            }
        }

        if (elempack == 1)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else
            {
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
                            space_ofs[p1] = p2;
                            p1++;
                            p2 += dilation_w;
                        }
                        p2 += gap;
                    }
                }

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < group; g++)
                {
                    __fp16* outptr = top_blob.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float sum = 0.f;

                            if (bias_term)
                                sum = bias_data[g];

                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                __fp16 val = sptr[space_ofs[k]];
                                __fp16 w = kptr[k];
                                sum += val * w;
                            }

                            sum = activation_ss(sum, activation_type, activation_params);

                            outptr[j] = (__fp16)sum;
                        }

                        outptr += outw;
                    }
                }
            }
        }

        return 0;
    }

    // group convolution
    const int channels_g = channels * elempack / group;
    const int num_output_g = num_output / group;

    int g_elempack = (opt.use_packing_layout && channels_g % packn == 0) ? packn : 1;
    int out_g_elempack = (opt.use_packing_layout && num_output_g % packn == 0) ? packn : 1;

    // unpacking
    Mat bottom_blob_bordered_unpacked = bottom_blob_bordered;
    if (elempack > g_elempack)
    {
        Option opt_p = opt;
        opt_p.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob_bordered, bottom_blob_bordered_unpacked, g_elempack, opt_p);
    }

    Mat top_blob_unpacked = top_blob;
    if (out_g_elempack < out_elempack)
    {
        top_blob_unpacked.create(outw, outh, num_output / out_g_elempack, out_elemsize / out_elempack * out_g_elempack, out_g_elempack, opt.workspace_allocator);
        if (top_blob_unpacked.empty())
            return -100;
    }

    for (int g = 0; g < group; g++)
    {
        const Mat bottom_blob_bordered_g = bottom_blob_bordered_unpacked.channel_range(channels_g * g / g_elempack, channels_g / g_elempack);
        Mat top_blob_g = top_blob_unpacked.channel_range(num_output_g * g / out_g_elempack, num_output_g / out_g_elempack);

        const ncnn::Layer* op = group_ops[g];

        Option opt_g = opt;
        opt_g.blob_allocator = top_blob_unpacked.allocator;

        // forward
        op->forward(bottom_blob_bordered_g, top_blob_g, opt_g);
    }

    // packing
    if (out_g_elempack < out_elempack)
    {
        convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
    }
    else
    {
        top_blob = top_blob_unpacked;
    }

    return 0;
}
#endif // __riscv_vector && __riscv_zfh

#if NCNN_INT8
int ConvolutionDepthWise_riscv::create_pipeline_int8(const Option& opt)
{
    int vl;
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        int elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            elempack = channels % 8 == 0 ? 8 : 1;
        }
#endif // __riscv_vector

        if (elempack == 8)
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, group);
            convert_packing(weight_data_r2, weight_data_tm, 8, opt);
        }

        if (elempack == 1)
        {
            weight_data_tm = weight_data;
        }

        weight_data.release();

        return 0;
    }

    // group convolution
    create_group_ops(opt);

    weight_data.release();

    return 0;
}

int ConvolutionDepthWise_riscv::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int vl;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;

    int elembits = bottom_blob.elembits();

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_int8 = bottom_blob;
    if (elembits != 8)
    {
        const int channels_g = channels * elempack / group;

        Mat scales(channels * elempack);
        {
            float* ps = scales;
            for (int g = 0; g < group; g++)
            {
                float scale = bottom_blob_int8_scales[g];
                for (int q = 0; q < channels_g; q++)
                {
                    *ps++ = scale;
                }
            }
        }

        Option opt_q = opt;
        opt_q.blob_allocator = opt.workspace_allocator;
        quantize_to_int8(bottom_blob, bottom_blob_int8, scales, opt_q);
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob_int8, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;
    channels = bottom_blob_bordered.c;
    elempack = bottom_blob_bordered.elempack;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    // depth-wise
    if (channels * elempack == group && group == num_output) // depth-wise conv, 逐通道卷积
    {
        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = num_output % 8 == 0 ? 8 : 1;
        }
#endif // __riscv_vector
        bool use_int8_requantize = int8_scale_term > 100;
        size_t out_elemsize = use_int8_requantize ? 1u * out_elempack : 4u * out_elempack;

        if (support_fp16_storage && opt.use_fp16_storage)
        {
            out_elemsize = use_int8_requantize ? 1u * out_elempack : 2u * out_elempack;
        }

        if (opt.use_bf16_storage)
            out_elemsize = use_int8_requantize ? 1u * out_elempack : 2u * out_elempack;

        top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        // TODO use fp16 / bf16
        out_elemsize = use_int8_requantize ? 1u * out_elempack : 4u * out_elempack;
        top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __riscv_vector
        if (elempack == 8)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (activation_type == 0 || activation_type == 1))
            {
                Mat top_blob_int32;
                top_blob_int32.create(outw, outh, num_output / out_elempack, (size_t)4u * out_elempack, out_elempack, opt.workspace_allocator);
                if (top_blob_int32.empty())
                    return -100;

                convdw3x3s1_pack8_int8_rvv(bottom_blob_bordered, top_blob_int32, weight_data_tm, opt);

                Mat scale_in_data(group);
                for (int g = 0; g < group; g++)
                {
                    // dequantize
                    float scale_in;
                    if (weight_data_int8_scales[g] == 0)
                        scale_in = 0;
                    else
                        scale_in = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

                    scale_in_data[g] = scale_in;
                }

                if (use_int8_requantize)
                {
                    requantize_from_int32_to_int8(top_blob_int32, top_blob, scale_in_data, top_blob_int8_scales, bias_data, activation_type, activation_params, opt);
                }
                else
                {
                    dequantize_from_int32(top_blob_int32, top_blob, scale_in_data, bias_data, opt);

                    if (activation)
                    {
                        activation->forward_inplace(top_blob, opt);
                    }
                }
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2 && (activation_type == 0 || activation_type == 1))
            {
                Mat top_blob_int32;
                top_blob_int32.create(outw, outh, num_output / out_elempack, (size_t)4u * out_elempack, out_elempack, opt.workspace_allocator);
                if (top_blob_int32.empty())
                    return -100;

                convdw3x3s2_pack8_int8_rvv(bottom_blob_bordered, top_blob_int32, weight_data_tm, opt);

                Mat scale_in_data(group);
                for (int g = 0; g < group; g++)
                {
                    // dequantize
                    float scale_in;
                    if (weight_data_int8_scales[g] == 0)
                        scale_in = 0;
                    else
                        scale_in = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

                    scale_in_data[g] = scale_in;
                }

                if (use_int8_requantize)
                {
                    requantize_from_int32_to_int8(top_blob_int32, top_blob, scale_in_data, top_blob_int8_scales, bias_data, activation_type, activation_params, opt);
                }
                else
                {
                    dequantize_from_int32(top_blob_int32, top_blob, scale_in_data, bias_data, opt);

                    if (activation)
                    {
                        activation->forward_inplace(top_blob, opt);
                    }
                }
            }
            else
            {
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
                            space_ofs[p1] = p2;
                            p1++;
                            p2 += dilation_w;
                        }
                        p2 += gap;
                    }
                }

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < channels; g++)
                {
                    signed char* outptr_s8 = top_blob.channel(g);
                    float* outptr_f32 = top_blob.channel(g);
                    const signed char* kptr = (const signed char*)weight_data_tm + maxk * g * 8;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            vl = 8;
                            vint32m2_t _sum0 = vmv_v_x_i32m2(0, vl);

                            const signed char* sptr = m.row<const signed char>(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++)
                            {
                                vint8m1_t _val = vle8_v_i8m1(sptr + space_ofs[k] * 8, vl);
                                vint8m1_t _w = vle8_v_i8m1(kptr + k * 8, vl);
                                vint16m1_t _s0 = vget_v_i16m2_i16m1(vwmul_vv_i16m2(_val, _w, vl), 0);

                                _sum0 = vwadd_wv_i32m2(_sum0, _s0, vl);
                            }

                            vfloat32m2_t _scale_in;
                            {
                                vfloat32m2_t _bottom_blob_int8_scales = vle32_v_f32m2((const float*)bottom_blob_int8_scales + g * 8, vl);
                                vfloat32m2_t _weight_data_int8_scales = vle32_v_f32m2((const float*)weight_data_int8_scales + g * 8, vl);

                                _scale_in = vfdiv_vv_f32m2(vfmv_v_f_f32m2(1.f, vl), vfmul_vv_f32m2(_bottom_blob_int8_scales, _weight_data_int8_scales, vl), vl);
                                vbool16_t _is_zero = vmfeq_vv_f32m2_b16(_bottom_blob_int8_scales, vfmv_v_f_f32m2(0.f, vl), vl);
                                _scale_in = vfsub_vv_f32m2_m(_is_zero, _scale_in, _scale_in, _scale_in, vl);
                            }

                            vfloat32m2_t _sumfp32 = vfmul_vv_f32m2(vfcvt_f_x_v_f32m2(_sum0, vl), _scale_in, vl);

                            if (bias_term)
                            {
                                vfloat32m2_t _bias = vle32_v_f32m2((const float*)bias_data + g * 8, vl);
                                _sumfp32 = vfadd_vv_f32m2(_sumfp32, _bias, vl);
                            }
                            _sumfp32 = activation_ps(_sumfp32, activation_type, activation_params, vl);

                            if (use_int8_requantize)
                            {
                                // requantize
                                vfloat32m2_t _scale_out = vle32_v_f32m2((const float*)top_blob_int8_scales + g * 8, vl);
                                vfloat32m2_t _res = vfmul_vv_f32m2(_sumfp32, _scale_out, vl);
                                int64_t _sum8 = float2int8(vget_v_f32m2_f32m1(_res, 0), vget_v_f32m2_f32m1(_res, 1));
                                *(int64_t*)outptr_s8 = _sum8;
                                outptr_s8 += 8;
                            }
                            else
                            {
                                // dequantize
                                vse32_v_f32m2(outptr_f32, _sumfp32, vl);
                                outptr_f32 += 8;
                            }
                        }
                    }
                }
            }
        }
#endif // __riscv_vector

        if (elempack == 1)
        {
            {
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
                            space_ofs[p1] = p2;
                            p1++;
                            p2 += dilation_w;
                        }
                        p2 += gap;
                    }
                }

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < group; g++)
                {
                    signed char* outptr_s8 = top_blob.channel(g);
                    float* outptr_f32 = top_blob.channel(g);
                    const signed char* kptr = (const signed char*)weight_data_tm + maxk * g;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            int sum = 0;

                            const signed char* sptr = m.row<const signed char>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                signed char val = sptr[space_ofs[k]];
                                signed char w = kptr[k];
                                sum += val * w;
                            }

                            float scale_in;
                            if (weight_data_int8_scales[g] == 0)
                                scale_in = 0;
                            else
                                scale_in = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

                            float sumfp32 = sum * scale_in;

                            if (bias_term)
                                sumfp32 += bias_data[g];

                            sumfp32 = activation_ss(sumfp32, activation_type, activation_params);

                            if (use_int8_requantize)
                            {
                                // requantize
                                float scale_out = top_blob_int8_scales[g];
                                signed char sums8 = float2int8(sumfp32 * scale_out);
                                outptr_s8[0] = sums8;
                                outptr_s8 += 1;
                            }
                            else
                            {
                                // dequantize
                                outptr_f32[0] = sumfp32;
                                outptr_f32 += 1;
                            }
                        }
                    }
                }
            }
        }

        return 0;
    }

    bool use_int8_requantize = int8_scale_term > 100;
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        if (use_int8_requantize)
            out_elempack = num_output % 8 == 0 ? 8 : 1;
        else
            out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __riscv_vector
    size_t out_elemsize = use_int8_requantize ? 1u * out_elempack : 4u * out_elempack;
    // #if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage)
    {
        out_elemsize = use_int8_requantize ? 1u * out_elempack : 2u * out_elempack;
    }
    // // #endif
    // if (opt.use_bf16_storage)
    //     out_elemsize = use_int8_requantize ? 1u * out_elempack : 2u * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // group convolution
    const int channels_g = channels * elempack / group;
    const int num_output_g = num_output / group;

    int g_elempack = 1;
    int out_g_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        g_elempack = channels_g % 8 == 0 ? 8 : 1;
        if (use_int8_requantize)
            out_g_elempack = num_output_g % 8 == 0 ? 8 : 1;
        else
            out_g_elempack = num_output_g % 4 == 0 ? 4 : 1;
    }
#endif // __riscv_vector

    // unpacking
    Mat bottom_blob_bordered_unpacked = bottom_blob_bordered;
    if (elempack > g_elempack)
    {
        Option opt_p = opt;
        opt_p.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob_bordered, bottom_blob_bordered_unpacked, g_elempack, opt_p);
    }

    Mat top_blob_unpacked = top_blob;
    if (out_g_elempack < out_elempack)
    {
        top_blob_unpacked.create(outw, outh, num_output / out_g_elempack, out_elemsize / out_elempack * out_g_elempack, out_g_elempack, opt.workspace_allocator);
        if (top_blob_unpacked.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        const Mat bottom_blob_bordered_g = bottom_blob_bordered_unpacked.channel_range(channels_g * g / g_elempack, channels_g / g_elempack);
        Mat top_blob_g = top_blob_unpacked.channel_range(num_output_g * g / out_g_elempack, num_output_g / out_g_elempack);

        const ncnn::Layer* op = group_ops[g];

        Option opt_g = opt;
        opt_g.blob_allocator = top_blob_unpacked.allocator;

        // forward
        op->forward(bottom_blob_bordered_g, top_blob_g, opt_g);
    }
    // packing
    if (out_g_elempack < out_elempack)
    {
        convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
    }
    else
    {
        top_blob = top_blob_unpacked;
    }

    return 0;
}
#endif // NCNN_INT8

} // namespace ncnn
