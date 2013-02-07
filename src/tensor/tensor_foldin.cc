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

#define TENSOR_LOAD_IMPL
#include <iostream>
#include <tensor/tensor.h>
#include <tensor/io.h>
#include <tensor/tensor_lapack.h>
#include "gemm.cc"

namespace tensor {

  using namespace blas;

  template<typename elt_t>
  void
  do_foldin_into(Tensor<elt_t> &output,
                 const Tensor<elt_t> &a, int _ndx1, const Tensor<elt_t> &b, int _ndx2)
  {
    index i_len,j_len,k_len,l_len,m_len;
    index rank, i;
    const index ranka = a.rank();
    const index rankb = b.rank();
    index ndx1 = normalize_index(_ndx1, ranka);
    index ndx2 = normalize_index(_ndx2, rankb);
    Indices new_dims(ranka + rankb - 2);
    /*
     * Since we use row-major order, in which the first
     * index varies faster, we nest the loops beginning with the last index,
     * and the loop what does is
     *		c(k,i,j,m) = a(i,l,j) * b(k,l,m)
     * where there is a sum over the repeated index "l". In the first part of
     * the code we find out the size of the contracted (l_len,l_len) and
     * uncontracted (new_dims, i_len,j_len,k_len,m_len) dimensions of the
     * tensors.
     */
    for (i = 0, rank = 0, k_len=1; i < ndx2; i++) {
      index di = b.dimension(i);
      new_dims.at(rank++) = di;
      k_len *= di;
    }
    l_len = b.dimension(i++);
    for (i = 0, i_len=1; i < ndx1; i++) {
      index di = a.dimension(i);
      new_dims.at(rank++) = di;
      i_len *= di;
    }
    if (l_len != a.dimension(i++)) {
      std::cerr << "Unable to foldin() tensors with dimensions" << std::endl
                << "\t" << a.dimensions() << " and "
                << b.dimensions() << std::endl
                << "\tbecause indices " << ndx1 << " and " << ndx2
                << " have different sizes" << std::endl;
      abort();
    }
    for (j_len = 1; i < ranka; i++) {
      index di = a.dimension(i);
      new_dims.at(rank++) = di;
      j_len *= di;
    }
    for (m_len = 1, i = ndx2+1; i < rankb; i++) {
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
    if (output.size() == 0)
      return;

    elt_t *pC = output.begin();
    elt_t zero = number_zero<elt_t>();
    elt_t one = number_one<elt_t>();
    const elt_t *pA = a.begin();
    const elt_t *pB = b.begin();
    char op1 = 'N';
    char op2 = 'T';
    index il_len = i_len*l_len;
    index kl_len = k_len*l_len;
    index ki_len = k_len*i_len;
    for (index m = 0; m < m_len; m++) {
      for (index j = 0; j < j_len; j++) {
        gemm(op1, op2, k_len, i_len, l_len, one,
             pB + kl_len*m, k_len, pA + il_len*j, i_len,
             zero, pC + ki_len*(j + j_len*m), k_len);
      }
    }
  }

} // namespace tensor
