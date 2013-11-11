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

#include <tensor/sparse.h>
#include "loops.h"


#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>

namespace tensor_test {

  template<typename elt_t>
  void test_sparse_index()
  {
    int i;
    Indices row, col;
    for (int N = 2; N < 3; N++) {
      Tensor<elt_t> aux = Tensor<elt_t>::random(N,N);
      for (int j = -N+1; j < N; j++) {
        Tensor<elt_t> diagonal = aux.diag(j);
        if (j >= 0) {
          row = iota(0, N-j-1);
          col = iota(j, N-1);
        } else {
          col = iota(0, N+j-1);
          row = iota(-j, N-1);
        }
        Sparse<elt_t> sparse(row, col, diagonal, N, N);
        Tensor<elt_t> exact = diag(diagonal, j, N, N);
        EXPECT_TRUE(all_equal(sparse, exact));
      }
    }
  }

  TEST(RSparseIndexTest, RSparseEmptyConstructor) {
    test_sparse_index<double>();
  }

  TEST(CSparseIndexTest, CSparseEmptyConstructor) {
    test_sparse_index<cdouble>();
  }

} // namespace tensor_test
