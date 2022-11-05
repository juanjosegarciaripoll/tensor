#pragma once
/*
    Copyright (c) 2010 Juan Jose Garcia Ripoll

    Tensor is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public License as published
    by the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Library General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/
#ifndef TENSOR_TENSOR_ACCUMULATE_H
#define TENSOR_TENSOR_ACCUMULATE_H

#include <tensor/indices.h>
#include <tensor/tensor/types.h>

namespace tensor {

struct AccumulateDimensions {
  const Dimensions output_dimensions;
  index left_size, size, right_size;
};

AccumulateDimensions prepare_accumulate(const Dimensions &dimensions,
                                        index axis);

template <typename elt, typename function>
inline Tensor<elt> accumulate(const Tensor<elt> &t, index axis, function f,
                              elt start = number_zero<elt>()) {
  auto data = prepare_accumulate(t.dimensions(), axis);
  auto output = Tensor<elt>::empty(Dimensions{data.left_size, data.right_size})
                    .fill_with(start);
  auto t3 =
      Tensor<elt>(Dimensions{data.left_size, data.size, data.right_size}, t);
  for (index k = 0; k < data.right_size; ++k) {
    for (index j = 0; j < data.size; ++j) {
      for (index i = 0; i < data.left_size; ++i) {
        auto &x = output.at(i, k);
        x = f(x, t3(i, j, k));
      }
    }
  }
  return Tensor<elt>(data.output_dimensions, output);
}

template <typename elt, typename function>
inline Tensor<elt> accumulate_no_start(const Tensor<elt> &t, index axis,
                                       function f) {
  auto data = prepare_accumulate(t.dimensions(), axis);
  auto t3 =
      Tensor<elt>(Dimensions{data.left_size, data.size, data.right_size}, t);
  Tensor<elt> output(t3(_, range(0), _).copy());
  for (index k = 0; k < data.right_size; ++k) {
    for (index j = 1; j < data.size; ++j) {
      for (index i = 0; i < data.left_size; ++i) {
        auto &x = output.at(i, k);
        x = f(x, t3(i, j, k));
      }
    }
  }
  return Tensor<elt>(data.output_dimensions, output);
}

template <typename elt>
inline Tensor<elt> sum(const Tensor<elt> &t, index axis) {
  return accumulate(t, axis, [](elt x, elt y) { return x + y; });
}

template <typename elt>
inline Tensor<elt> mean(const Tensor<elt> &t, index axis) {
  auto &d = t.dimensions();
  elt L = static_cast<elt>(d[Dimensions::normalize_index(axis, d.rank())]);
  return accumulate(t, axis, [L](elt x, elt y) -> elt { return x + y / L; });
}

template <typename elt>
inline Tensor<elt> max(const Tensor<elt> &t, index axis) {
  return accumulate_no_start(
      t, axis, [](elt x, elt y) -> elt { return std::max(x, y); });
}

template <typename elt>
inline Tensor<elt> min(const Tensor<elt> &t, index axis) {
  return accumulate_no_start(
      t, axis, [](elt x, elt y) -> elt { return std::min(x, y); });
}

}  // namespace tensor

#endif  // TENSOR_TENSOR_ACCUMULATE_H