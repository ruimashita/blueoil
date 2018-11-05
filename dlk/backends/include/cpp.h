/* Copyright 2018 The Blueoil Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once
#include "common/global.h"

namespace cpp {
using namespace type;
namespace p_3x3 = conv3x3_params;
namespace p_1x1 = conv1x1_params;

/// pure convolution with 1x1 and 3x3 kernel
/// @{
void conv3x3_impl(T_in in_data[], T_out out_data[], T_k k_data[],
                  T_out threshold_data[], unsigned in_w, unsigned in_h,
                  unsigned in_c, unsigned out_w, unsigned out_h, unsigned out_c,
                  unsigned pad, unsigned stride);

void qconv3x3_impl(T_q in_data[], T_out out_data[], T_q k_data[], unsigned in_w,
                   unsigned in_h, unsigned in_c_by_word, unsigned out_w,
                   unsigned out_h, unsigned out_c, unsigned out_c_offset,
                   unsigned pad, unsigned stride);

void conv1x1_impl(T_in in_data[], T_out out_data[], T_k k_data[],
                  T_out threshold_data[], unsigned in_w, unsigned in_h,
                  unsigned in_c, unsigned out_c);

void qconv1x1_impl(T_q in_data_packed[], T_out out_data[], T_q k_data_packed[],
                   unsigned in_w, unsigned in_h, unsigned in_c_by_word,
                   unsigned out_c);

template <int KH, int KW>
void conv(T_in in_data[], T_out out_data[], T_k k_data[],
          T_out threshold_data[], unsigned in_w, unsigned in_h, unsigned in_c,
          unsigned out_w, unsigned out_h, unsigned out_c, unsigned pad,
          unsigned stride) {
  assert(((KH == 3) && (KW == 3)) || ((KH == 1) && (KW == 1)));
  assert(((KH == 3) && (pad == 1)) || ((KH == 1) && (pad == 0)));
  assert(stride == 1);

  if (KH == 3 && KW == 3) {
    cpp::conv3x3_impl(in_data, out_data, k_data, threshold_data, in_w, in_h,
                      in_c, out_w, out_h, out_c, pad, stride);
  } else if (KH == 1 && KW == 1) {
    cpp::conv1x1_impl(in_data, out_data, k_data, threshold_data, in_w, in_h,
                      in_c, out_c);
  }
}
/// @}

/// convolution with kn2row and tiling
/// @{
void conv_kn2row_tiling_impl(T_in in_data[], T_out out_data[], T_k k_data[],
                             T_out threshold_data[], unsigned in_w,
                             unsigned in_h, unsigned in_c, unsigned out_w,
                             unsigned out_h, unsigned out_c, unsigned k_w,
                             unsigned k_h, unsigned pad, unsigned stride);

template <int KH, int KW>
void conv_kn2row_tiling(T_in in_data[], T_out out_data[], T_k k_data[],
                        T_out threshold_data[], unsigned in_w, unsigned in_h,
                        unsigned in_c, unsigned out_w, unsigned out_h,
                        unsigned out_c, unsigned pad, unsigned stride) {
  assert(((KH == 3) && (KW == 3)) || ((KH == 1) && (KW == 1)));
  assert(((KH == 3) && (pad == 1)) || ((KH == 1) && (pad == 0)));
  assert(stride == 1);

  cpp::conv_kn2row_tiling_impl(in_data, out_data, k_data, threshold_data, in_w,
                               in_h, in_c, out_w, out_h, out_c, KW, KW, pad,
                               stride);
}
/// @}

/// 2A 1W quantized convolution with 1x1 and 3x3 kernel
template <int KH, int KW>
void qconv(T_q in_data_packed[], T_out out_data[], T_q k_data_packed[],
           unsigned in_w, unsigned in_h, unsigned in_c_by_word,
           unsigned nbits_in_data, unsigned out_w, unsigned out_h,
           unsigned out_c, unsigned pad, unsigned stride) {
  assert(((KH == 3) && (KW == 3)) || ((KH == 1) && (KW == 1)));
  assert(((KH == 3) && (pad == 1)) || ((KH == 1) && (pad == 0)));
  assert(stride == 1);

  if (KH == 3 && KW == 3) {
    cpp::qconv3x3_impl(in_data_packed, out_data, k_data_packed, in_w, in_h,
                       in_c_by_word, out_w, out_h, out_c, 0, pad, stride);
  } else if (KH == 1 && KW == 1) {
    cpp::qconv1x1_impl(in_data_packed, out_data, k_data_packed, in_w, in_h,
                       in_c_by_word, out_c);
  }
}

template <class Ta, class Tb, class Ty>
void gemm(Ta A[], Tb B[], Ty Y[], u32 a_row, u32 a_col, u32 b_col);

template <class Tx, class Ty, int Nbit_a, int Nbit_b>
void qgemm(Tx A_packed[], Tx B_packed[], Ty Y[], u32 a_row, u32 a_col,
           u32 b_col);

void qconv_with_kn2row_impl(T_q in_data[], T_out out_data[], T_q k_data[],
                            T_out partial_out_data[], T_out threshold_data[],
                            unsigned in_w, unsigned in_h, unsigned in_c,
                            unsigned out_w, unsigned out_h, unsigned out_c,
                            unsigned k_w, unsigned k_h, unsigned pad);

template <int KH, int KW>
void qconv_with_kn2row(T_q in_data_packed[], T_out out_data[],
                       T_q k_data_packed[], T_out threshold_data[],
                       unsigned in_w, unsigned in_h, unsigned in_c_by_word,
                       unsigned nbits_in_data, unsigned out_w, unsigned out_h,
                       unsigned out_c, unsigned pad, unsigned stride) {
  assert(((KH == 3) && (KW == 3)) || ((KH == 1) && (KW == 1)));
  assert(((KH == 3) && (pad == 1)) || ((KH == 1) && (pad == 0)));
  assert(stride == 1);

  qconv_with_kn2row_impl(in_data_packed, out_data, k_data_packed, out_data,
                         threshold_data, in_w, in_h, in_c_by_word, out_w, out_h,
                         out_c, KH, KW, pad);
}

void qconv_kn2row_tiling_impl(T_q in_data[], T_out out_data[], T_q k_data[],
                              T_out partial_out_data[], T_out threshold_data[],
                              unsigned in_w, unsigned in_h, unsigned in_c,
                              unsigned out_w, unsigned out_h, unsigned out_c,
                              unsigned k_w, unsigned k_h, unsigned pad);

template <int KH, int KW>
void qconv_kn2row_tiling(T_q in_data_packed[], T_out out_data[],
                         T_q k_data_packed[], T_out threshold_data[],
                         unsigned in_w, unsigned in_h, unsigned in_c_by_word,
                         unsigned nbits_in_data, unsigned out_w, unsigned out_h,
                         unsigned out_c, unsigned pad, unsigned stride) {
  assert(((KH == 3) && (KW == 3)) || ((KH == 1) && (KW == 1)));
  assert(((KH == 3) && (pad == 1)) || ((KH == 1) && (pad == 0)));
  assert(stride == 1);

  qconv_kn2row_tiling_impl(in_data_packed, out_data, k_data_packed, out_data,
                           threshold_data, in_w, in_h, in_c_by_word, out_w,
                           out_h, out_c, KH, KW, pad);
}
/// @}

void a8w1_conv3x3_impl(T_in_k2c in_data[], T_out_k2c out_data[],
                       T_k_k2c k_data[], unsigned in_w, unsigned in_h,
                       unsigned in_c, unsigned out_w, unsigned out_h,
                       unsigned out_c, unsigned pad, unsigned stride);

void a8w1_conv3x3_with_kn2row_impl(T_in_k2c in_data[], T_out_k2c out_data[],
                                   T_k_k2c k_data[],
                                   T_out_k2c out_data_partial[], unsigned in_w,
                                   unsigned in_h, unsigned in_c, unsigned out_w,
                                   unsigned out_h, unsigned out_c,
                                   unsigned pad);

template <int KH, int KW>
void a8w1_conv(T_in_k2c in_data[], T_out_k2c out_data[], T_k_k2c k_data[],
               unsigned in_w, unsigned in_h, unsigned in_c, unsigned out_w,
               unsigned out_h, unsigned out_c, unsigned pad, unsigned stride) {
  assert((KH == 3) && (KW == 3));
  assert(((KH == 3) && (pad == 1)) || ((KH == 1) && (pad == 0)));
  assert(stride == 1);

  if (KH == 3 && KW == 3) {
    a8w1_conv3x3_impl(in_data, out_data, k_data, in_w, in_h, in_c, out_w, out_h,
                      out_c, pad, stride);
  }
}

template <int KH, int KW>
void a8w1_conv3x3_with_kn2row(T_in_k2c in_data[], T_out_k2c out_data[],
                              T_k_k2c k_data[], unsigned in_w, unsigned in_h,
                              unsigned in_c, unsigned out_w, unsigned out_h,
                              unsigned out_c, unsigned pad, unsigned stride) {
  assert((KH == 3) && (KW == 3));
  assert(((KH == 3) && (pad == 1)) || ((KH == 1) && (pad == 0)));
  assert(stride == 1);

  if (KH == 3 && KW == 3) {
    a8w1_conv3x3_with_kn2row_impl(in_data, out_data, k_data, out_data, in_w,
                                  in_h, in_c, out_w, out_h, out_c, pad);
  }
}
} // namespace cpp
