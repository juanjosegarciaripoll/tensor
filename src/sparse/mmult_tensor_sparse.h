// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
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

//////////////////////////////////////////////////////////////////////
// RAW ROUTINE FOR THE TENSOR-SPARSE PRODUCT
//

template <typename elt_t>
static void mult_t_sp(elt_t *dest, const elt_t *vector, const index *row_start,
                      const index *column, const elt_t *matrix, index i_len,
                      index j_len, index k_len, index l_len) {
  // dest(i,k,l) = vector(i,j,k) matrix(j,l)
  for (index j = 0; j < j_len; j++, vector += i_len) {
    for (index x = row_start[j]; x < row_start[j + 1]; x++) {
      index l = column[x];
      elt_t *d = dest + l * (k_len * i_len);
      const elt_t *v = vector;
      elt_t m = matrix[x];
      for (index k = 0; k < k_len; k++) {
        for (index i = 0; i < i_len; i++, d++) {
          *d += *(v++) * m;
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
static inline const Tensor<elt_t> do_mmult(const Tensor<elt_t> &m1,
                                           const Sparse<elt_t> &m2) {
  index N = m1.rank();
  index i_len = 1;
  Indices dims(N);
  for (index k = 0; k < N - 1; k++) {
    dims.at(k) = m1.dimension(k);
    i_len *= dims[k];
  }
  index j_len = m1.dimension(-1);
  index l_len = dims.at(N - 1) = m2.columns();

  if (j_len != m2.rows()) {
    std::cerr << "In mmult(T,S), the last index of tensor T does not match the "
                 "number of rows\n"
                 "in sparse matrix S.";
    abort();
  }

  Tensor<elt_t> output = Tensor<elt_t>::zeros(dims);

  mult_t_sp<elt_t>(output.begin(), m1.begin(), m2.priv_row_start().begin(),
                   m2.priv_column().begin(), m2.priv_data().begin(), i_len,
                   j_len, 1, l_len);

  return output;
}

#endif /* TENSOR_MMULT_TENSOR_SPARSE_H */
