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

#include <algorithm>
#include <tensor/tensor.h>

template <typename elt_t>
static void do_diag(elt_t *output, const elt_t *input, tensor::index a1,
                    tensor::index a2, tensor::index a2b, tensor::index a3,
                    tensor::index a4, tensor::index a5, tensor::index which) {
  // output(a1,a2b,a3,a5), input(a1,a2,a3,a4,a5)
  tensor::index o1, o2;
  if (which < 0) {
    o2 = -which;
    o1 = 0;
  } else {
    o2 = 0;
    o1 = which;
  }
  for (tensor::index m = 0; m < a5; m++) {
    for (tensor::index l = 0; l < a2b; l++) {
      for (tensor::index k = 0; k < a3; k++) {
        for (tensor::index i = 0; i < a1; i++) {
          output[i + a1 * (l + a2b * (k + a3 * m))] =
              input[i + a1 * ((o1 + l) + a2 * (k + a3 * ((o2 + l) + a4 * m)))];
        }
      }
    }
  }
}

/* Extract a diagonal from a matrix. */
template <typename elt_t>
const tensor::Tensor<elt_t> do_take_diag(const tensor::Tensor<elt_t> &a,
                                         tensor::index which,
                                         tensor::index ndx1,
                                         tensor::index ndx2) {
  if (ndx1 < 0) ndx1 += a.rank();
  tensor_assert((ndx1 < a.rank()) && (ndx1 >= 0));
  if (ndx2 < 0) ndx2 += a.rank();
  tensor_assert((ndx2 < a.rank()) && (ndx2 >= 0));

  tensor::index new_rank = std::max(a.rank() - 1, tensor::index(1));
  tensor::Indices new_dims(new_rank);
  tensor::index i, rank = 0;
  tensor::index a1, a2, a3, a4, a5, a2b;
  if (ndx1 > ndx2) {
    std::swap(ndx1, ndx2);
    which = -which;
  }
  for (i = 0, a1 = 1; i < ndx1; i++) {
    auto di = a.dimension(i);
    new_dims.at(rank++) = di;
    a1 *= di;
  }
  a2 = a.dimension(i++);
  new_dims.at(rank++) = a2;
  for (a3 = 1; i < ndx2; i++) {
    auto di = a.dimension(i);
    new_dims.at(rank++) = di;
    a3 *= di;
  }
  a4 = a.dimension(i++);
  for (a5 = 1; i < a.rank(); i++) {
    auto di = a.dimension(i);
    new_dims.at(rank++) = di;
    a5 *= di;
  }
  if (which <= -a2 || which >= a4) {
    std::cerr << "In take_diag(M, which, ...), WHICH has a value " << which
              << " which exceeds the size of the tensor";
    abort();
  }
  if (a2 == 1 && a4 == 1) {
    return reshape(a, new_dims);
  }
  if (which < 0) {
    a2b = std::max(tensor::index(0), std::min(a2 + which, a4));
  } else {
    a2b = std::max(tensor::index(0), std::min(a2, a4 - which));
  }
  new_dims.at(ndx1) = a2b;
  tensor::Tensor<elt_t> output(new_dims);
  which = -which;
  if (a2b) {
    do_diag<elt_t>(output.begin(), a.begin(), a1, a2, a2b, a3, a3, a5, which);
  }
  return output;
}
