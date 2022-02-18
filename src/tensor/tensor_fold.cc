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

#define TENSOR_LOAD_IMPL
#include <tensor/exceptions.h>
#include <tensor/tensor.h>
#include <tensor/tensor_lapack.h>
#include "gemm.cc"

namespace tensor {

using namespace blas;

template <typename elt_t, bool do_conj>
void do_fold(Tensor<elt_t> &output, const Tensor<elt_t> &a, int _ndx1,
             const Tensor<elt_t> &b, int _ndx2) {
  index i_len, j_len, k_len, l_len, m_len;
  int rank, i;
  const index ranka = a.rank();
  const index rankb = b.rank();
  index ndx1 = Dimensions::normalize_index(_ndx1, ranka);
  index ndx2 = Dimensions::normalize_index(_ndx2, rankb);
  Indices new_dims(static_cast<size_t>(std::max(ranka + rankb - 2, index(1))));
  /*
     * Since we use row-major order, in which the first
     * index varies faster, we nest the loops beginning with the last index,
     * and the loop what does is
     *		c(i,j,k,m) = a(i,l,j) * b(k,l,m)
     * where there is a sum over the repeated index "l". In the first part of
     * the code we find out the size of the contracted (l_len,l_len) and
     * uncontracted (new_dims, i_len,j_len,k_len,m_len) dimensions of the
     * tensors.
     */
  for (i = 0, rank = 0, i_len = 1; i < ndx1; i++) {
    index di = a.dimension(i);
    new_dims.at(rank++) = di;
    i_len *= di;
  }
  l_len = a.dimension(i++);
  if (l_len == 0) {
    throw dimensions_mismatch(a.dimensions(), b.dimensions(), ndx1, ndx2);
  }
  for (j_len = 1; i < ranka; i++) {
    index di = a.dimension(i);
    new_dims.at(rank++) = di;
    j_len *= di;
  }
  for (i = 0, k_len = 1; i < ndx2; i++) {
    index di = b.dimension(i);
    new_dims.at(rank++) = di;
    k_len *= di;
  }
  if (l_len != b.dimension(i++)) {
    throw dimensions_mismatch(a.dimensions(), b.dimensions(), ndx1, ndx2);
  }
  for (m_len = 1; i < rankb; i++) {
    index di = b.dimension(i);
    new_dims.at(rank++) = di;
    m_len *= di;
  }
  /*
     * Create the output tensor. Sometimes it is just a number.
     */
  if (rank == 0) {
    rank = 1;
    new_dims.at(0) = 1;
  }
  output = Tensor<elt_t>(new_dims);
  if (output.size() == 0) return;

  elt_t *pC = output.begin();
  const elt_t zero = number_zero<elt_t>();
  const elt_t one = number_one<elt_t>();
  const elt_t *pA = a.begin();
  const elt_t *pB = b.begin();
  if (i_len == 1) {
    if (k_len == 1) {
      // C(j_len,m_len) = A(l_len,j_len)*B(l_len,m_len);
      char transa = do_conj ? 'C' : 'T';
      char transb = 'N';
      gemm(transa, transb, j_len, m_len, l_len, one, pA, l_len, pB, l_len, zero,
           pC, j_len);
      return;
    }
    if (m_len == 1) {
      // C(j_len,k_len) = A(l_len,j_len)*B(k_len,l_len);
      char transa = do_conj ? 'C' : 'T';
      char transb = 'T';
      gemm(transa, transb, j_len, k_len, l_len, one, pA, l_len, pB, k_len, zero,
           pC, j_len);
      return;
    }
  } else if (j_len == 1 && !do_conj) {
    if (k_len == 1) {
      // C(i_len,m_len) = A(i_len,l_len)*B(l_len,m_len);
      char transa = 'N';
      char transb = 'N';
      gemm(transa, transb, i_len, m_len, l_len, one, pA, i_len, pB, l_len, zero,
           pC, i_len);
      return;
    }
    if (m_len == 1) {
      // C(i_len,k_len) = A(i_len,l_len)*B(k_len,l_len);
      char transa = 'N';
      char transb = 'T';
      gemm(transa, transb, i_len, k_len, l_len, one, pA, i_len, pB, k_len, zero,
           pC, i_len);
      return;
    }
  }
  const char op1 = 'N';
  const char op2 = do_conj ? 'C' : 'T';
  const index ij_len = i_len * j_len;
  const index il_len = i_len * l_len;
  const index kl_len = k_len * l_len;
  const index jk_len = j_len * k_len;
  /*
     * C(i,j,k,m) = A(i,l,j) * B(k,l,m)
     */
  for (index m = 0; m < m_len; m++) {
    for (index j = 0; j < j_len; j++) {
      gemm(op1, op2, i_len, k_len, l_len, one, pA + il_len * j, i_len,
           pB + kl_len * m, k_len, zero, pC + i_len * (j + jk_len * m), ij_len);
    }
  }
  if (do_conj) {
    for (index j = output.ssize(); j; j--, pC++) *pC = tensor::conj(*pC);
  }
}

}  // namespace tensor
