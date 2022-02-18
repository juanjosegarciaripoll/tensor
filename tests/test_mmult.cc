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

#include "loops.h"
#include <gtest/gtest.h>
#include <tensor/exceptions.h>
#include <tensor/tensor.h>

#include "slow_mmult.cc"

namespace tensor_test {

//////////////////////////////////////////////////////////////////////
// MATRIX MULTIPLICATION
//

template <typename n1, typename n2>
void test_mmult(index max_dim) {
  for (index i = 1; i <= max_dim; i++) {
    for (index j = 1; j <= max_dim; j++) {
      for (index k = 1; k <= max_dim; k++) {
        auto A = Tensor<n1>::random(i, j);
        auto B = Tensor<n1>::random(j, k);
        EXPECT_TRUE(approx_eq(mmult(A, B), fold_22_12(A, B)));
        unique(A);
        unique(B);
      }
    }
  }
  std::cout << std::endl;
  ASSERT_THROW(mmult(Tensor<n1>::eye(1, 0), Tensor<n2>::ones(0, 3)),
               dimensions_mismatch);
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

#define MATRIX_MAX_DIM 24

TEST(MmultTest, MmultDoubleDoubleTest) {
  test_mmult<double, double>(MATRIX_MAX_DIM);
}

TEST(MmultTest, MmultCdoubleCdoubleTest) {
  test_mmult<cdouble, cdouble>(MATRIX_MAX_DIM);
}

}  // namespace tensor_test
