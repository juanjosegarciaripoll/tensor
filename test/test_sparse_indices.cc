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
    for (int i = 2; i < 3; i++) {
      Tensor<elt_t> aux = Tensor<elt_t>::random(i,i);
      for (int j = -i+1; j < i; j++) {
	Tensor<elt_t> d = aux.diag(j);
	if (j >= 0) {
	  row = iota(0, i-j-1);
	  col = iota(j, i-1);
	} else {
	  col = iota(0, i+j-1);
	  row = iota(-j, i-1);
	}
	Sparse<elt_t> sparse(igen << i << i, row, col, d);
	Tensor<elt_t> exact = diag(d, j, i, i);
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
