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

using namespace tensor;

template <typename elt_t>
static void do_diag(elt_t *output, const elt_t *input, index_t a1, index_t a2,
                    index_t a2b, index_t a3, index_t a4, index_t a5,
                    index_t which) {
  // output(a1,a2b,a3,a5), input(a1,a2,a3,a4,a5)
  index_t o1{0}, o2{0};
  if (which < 0) {
    o2 = -which;
  } else {
    o1 = which;
  }
  for (index_t m = 0; m < a5; m++) {
    for (index_t l = 0; l < a2b; l++) {
      for (index_t k = 0; k < a3; k++) {
        for (index_t i = 0; i < a1; i++) {
          output[i + a1 * (l + a2b * (k + a3 * m))] =
              input[i + a1 * ((o1 + l) + a2 * (k + a3 * ((o2 + l) + a4 * m)))];
        }
      }
    }
  }
}

/* Extract a diagonal from a matrix. */
template <typename elt_t>
const Tensor<elt_t> do_take_diag(const Tensor<elt_t> &a, index_t which,
                                 index_t ndx1, index_t ndx2) {
  if (ndx1 < 0) ndx1 += a.rank();
  tensor_assert((ndx1 < a.rank()) && (ndx1 >= 0));
  if (ndx2 < 0) ndx2 += a.rank();
  tensor_assert((ndx2 < a.rank()) && (ndx2 >= 0));

  index_t new_rank = std::max(a.rank() - 1, index_t(1));
  Indices new_dims(new_rank);
  index_t i{0}, rank = 0;
  index_t a1{1}, a3{1}, a5{1};
  if (ndx1 > ndx2) {
    std::swap(ndx1, ndx2);
    which = -which;
  }
  for (i = 0, a1 = 1; i < ndx1; i++) {
    auto di = a.dimension(i);
    new_dims.at(rank++) = di;
    a1 *= di;
  }
  index_t a2 = a.dimension(i++);
  new_dims.at(rank++) = a2;
  for (a3 = 1; i < ndx2; i++) {
    auto di = a.dimension(i);
    new_dims.at(rank++) = di;
    a3 *= di;
  }
  index_t a4 = a.dimension(i++);
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
  index_t a2b = std::max<index_t>(
      0, which < 0 ? std::min(a2 + which, a4) : std::min(a2, a4 - which));
  new_dims.at(ndx1) = a2b;
  auto output = Tensor<elt_t>::empty(new_dims);
  which = -which;
  if (a2b) {
    do_diag<elt_t>(output.unsafe_begin_not_shared(), a.begin(), a1, a2, a2b, a3,
                   a3, a5, which);
  }
  return output;
}
