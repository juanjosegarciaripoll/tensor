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

#ifndef TENSOR_MMULT_TENSOR_SPARSE_H
#define TENSOR_MMULT_TENSOR_SPARSE_H

#include <algorithm>
#include <tensor/tensor.h>
#include <tensor/sparse.h>

namespace tensor {

//////////////////////////////////////////////////////////////////////
// RAW ROUTINE FOR THE TENSOR-SPARSE PRODUCT
//

// TODO: Add additional l_len for multiplication

template <typename elt_t>
static void mult_t_sp(elt_t *dest, const elt_t *vector,
                      const index_t *row_start, const index_t *column,
                      const elt_t *matrix, index_t i_len, index_t j_len,
                      index_t k_len, index_t /*l_len*/) {
  // dest(i,k,l) = vector(i,j,k) matrix(j,l)
  for (index_t j = 0; j < j_len; j++, vector += i_len) {
    for (index_t x = row_start[j]; x < row_start[j + 1]; ++x) {
      const auto l = column[x];
      const auto m = matrix[x];
      elt_t *d = dest + l * (k_len * i_len);
      const elt_t *v = vector;
      for (index_t k = 0; k < k_len; ++k) {
        for (index_t i = 0; i < i_len; ++i, ++d, ++v) {
          *d += *v * m;
        }
        v += (j_len - 1) * i_len;
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////
// HIGHER LEVEL INTERFACE
//

template <typename elt_t>
static bool matching_mmult_dimensions(const Tensor<elt_t> &m1,
                                      const Sparse<elt_t> &m2) {
  return m1.dimension(m1.rank() - 1) == m2.rows();
}

template <typename elt_t>
static inline Tensor<elt_t> do_mmult(const Tensor<elt_t> &m1,
                                     const Sparse<elt_t> &m2) {
  tensor_assert(matching_mmult_dimensions(m1, m2));

  index_t N = m1.rank();
  Indices dims(N);
  std::copy(m1.dimensions().begin(), m1.dimensions().end(), dims.begin());
  index_t j_len = m2.rows();
  index_t k_len = 1;
  index_t l_len = dims.at(N - 1) = m2.columns();
  index_t i_len = m1.ssize() / (k_len * j_len);

  Tensor<elt_t> output = Tensor<elt_t>::zeros(dims);

  mult_t_sp<elt_t>(output.unsafe_begin_not_shared(), m1.cbegin(),
                   m2.priv_row_start().cbegin(), m2.priv_column().cbegin(),
                   m2.priv_data().cbegin(), i_len, j_len, k_len, l_len);

  return output;
}

template <typename elt_t>
static bool matching_mmult_into_dimensions(Tensor<elt_t> &output,
                                           const Tensor<elt_t> &m1,
                                           const Sparse<elt_t> &m2) {
  auto rank = m1.rank();
  if (rank == output.rank()) {
    if (output.dimension(rank - 1) == m2.columns()) {
      return std::equal(output.dimensions().begin(),
                        output.dimensions().end() - 1, m1.dimensions().begin());
    }
  }
  return false;
}

template <typename elt_t>
static inline void do_mmult_into(Tensor<elt_t> &output, const Tensor<elt_t> &m1,
                                 const Sparse<elt_t> &m2) {
  tensor_assert(matching_mmult_dimensions(m1, m2));
  tensor_assert(matching_mmult_into_dimensions(output, m1, m2));

  index_t j_len = m2.rows();
  index_t k_len = 1;
  index_t l_len = m2.columns();
  index_t i_len = m1.ssize() / (k_len * j_len);

  output.fill_with_zeros();

  mult_t_sp<elt_t>(output.begin(), m1.cbegin(), m2.priv_row_start().cbegin(),
                   m2.priv_column().cbegin(), m2.priv_data().cbegin(), i_len,
                   j_len, k_len, l_len);
}

}  // namespace tensor

#endif /* TENSOR_MMULT_TENSOR_SPARSE_H */
