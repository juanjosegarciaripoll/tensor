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

#ifndef TENSOR_MMULT_SPARSE_TENSOR_H
#define TENSOR_MMULT_SPARSE_TENSOR_H

#include <algorithm>
#include <tensor/tensor.h>
#include <tensor/sparse.h>

namespace tensor {

//////////////////////////////////////////////////////////////////////
// RAW ROUTINES FOR THE SPARSE-TENSOR PRODUCT
//

template <typename elt_t>
static void mult_sp_t(elt_t *dest, const index_t *row_start,
                      const index_t *column, const elt_t *matrix,
                      const elt_t *vector, index_t i_len, index_t j_len,
                      index_t k_len, index_t l_len) {
  if (k_len == 1) {
#if 0
	// dest(i,l) = matrix(i,j) vector(j,l)
	for (; l_len; l_len--, vector+=j_len) {
	    for (index_t i = 0; i < i_len; i++) {
		elt_t accum = *dest;
		for (index_t j = row_start[i]; j < row_start[i+1]; j++) {
		    accum += matrix[j] * vector[column[j]];
		}
		*(dest++) = accum;
	    }
	}
#else
    for (; l_len; l_len--, vector += j_len) {
      const elt_t *m = matrix;
      const index_t *c = column;
      for (index_t i = 0; i < i_len; i++) {
        elt_t accum = *dest;
        for (index_t j = row_start[i + 1] - row_start[i]; j; j--) {
          accum += *(m++) * vector[*(c++)];
        }
        *(dest++) = accum;
      }
    }
#endif
  } else {
    // dest(i,k,l) = matrix(i,j) vector(k,j,l)
    for (index_t l = 0; l < l_len; l++) {
      const elt_t *v = vector + l * (k_len * j_len);
      for (index_t k = 0; k < k_len; k++, v++) {
        for (index_t i = 0; i < i_len; i++) {
          elt_t accum = *dest;
          for (index_t j = row_start[i]; j < row_start[i + 1]; j++) {
            accum += matrix[j] * v[column[j] * k_len];
          }
          *(dest++) = accum;
        }
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////
// HIGHER LEVEL INTERFACE
//

template <typename elt_t>
static inline Tensor<elt_t> do_mmult(const Sparse<elt_t> &m1,
                                     const Tensor<elt_t> &m2) {
  tensor_assert(m1.columns() == m2.dimension(0));

  Indices dims(m2.rank());
  std::copy(m2.dimensions().begin(), m2.dimensions().end(), dims.begin());
  index_t i_len = dims.at(0) = m1.rows();
  index_t k_len = 1;
  index_t j_len = m2.dimension(0);
  index_t l_len = m2.ssize() / j_len;

  auto output = Tensor<elt_t>::zeros(dims);

  mult_sp_t(output.unsafe_begin_not_shared(), m1.priv_row_start().cbegin(),
            m1.priv_column().cbegin(), m1.priv_data().cbegin(), m2.cbegin(),
            i_len, j_len, k_len, l_len);

  return output;
}

template <typename elt_t>
static inline void do_mmult_into(Tensor<elt_t> &output, const Sparse<elt_t> &m1,
                                 const Tensor<elt_t> &m2) {
  tensor_assert(output.rank() == m2.rank());
  tensor_assert(output.dimension(0) == m1.rows(0));
  tensor_assert(m2.dimension(0) == m2.columns(0));
  tensor_assert(std::equal(output.dimensions().begin() + 1,
                           output.dimensions.end(),
                           m2.dimensions().begin() + 1));

  output.fill_with_zeros();
  index_t i_len = m1.rows();
  index_t k_len = 1;
  index_t j_len = m2.dimension(0);
  index_t l_len = m2.ssize() / j_len;
  mult_sp_t(output.begin(), m1.priv_row_start().cbegin(),
            m1.priv_column().cbegin(), m1.priv_data().cbegin(), m2.cbegin(),
            i_len, j_len, k_len, l_len);
}

}  // namespace tensor

#endif /* !TENSOR_MMULT_SPARSE_TENSOR_H */
