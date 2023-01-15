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
void test_mmult(index_t max_dim, double density = 0.1) {
  for (index_t i = 1; i <= max_dim; i++) {
    for (index_t j = 1; j <= max_dim; j++) {
      for (index_t k = 1; k <= max_dim; k++) {
        auto A = Sparse<n1>::random(i, j, density);
        auto B = Tensor<n1>::random(j, k);
        auto C = mmult(A, B);
        EXPECT_CEQ(C, mmult(full(A), B));
        auto D = Tensor<n1>::empty(i, k);
        EXPECT_CEQ(mmult(A, B), (mmult_into(D, A, B), D));
        unique(B);
        unique(D);
      }
    }
  }
  ASSERT_THROW_DEBUG(mmult(Tensor<n1>::eye(1, 0), Tensor<n2>::ones(0, 3)),
                     dimensions_mismatch);
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

static constexpr int MATRIX_MAX_DIM = 24;

TEST(RSparseMMultTest, SmallMatrices) {
  {
    auto A = RSparse::eye(2);
    auto B = RTensor({{2.0, -3.0}, {-1.0, 4.0}});
    EXPECT_ALL_EQUAL(mmult(A, B), B);

    auto D = RTensor::empty(2, 2);
    mmult_into(D, A, B);
    EXPECT_ALL_EQUAL(D, B);
  }
  {
    auto A = RSparse({{1.0, 0.0}, {0.0, 2.0}});
    auto B = RTensor({{2.0, -3.0}, {-1.0, 4.0}});
    auto C = RTensor({{2.0, -3.0}, {-2.0, 8.0}});
    EXPECT_ALL_EQUAL(mmult(A, B), C);

    auto D = RTensor::empty(2, 2);
    mmult_into(D, A, B);
    EXPECT_ALL_EQUAL(D, C);
  }
  {
    auto A = RSparse({{1.0, -1.0}, {0.0, 2.0}});
    auto B = RTensor({{2.0, -3.0}, {-1.0, 4.0}});
    auto C = RTensor({{3.0, -7.0}, {-2.0, 8.0}});
    EXPECT_ALL_EQUAL(mmult(A, B), C);

    auto D = RTensor::empty(2, 2);
    mmult_into(D, A, B);
    EXPECT_ALL_EQUAL(D, C);
  }
}

TEST(RSparseMMultTest, HigherRankTensors) {
  {
    auto A = RSparse::eye(2);
    auto B = RTensor::random(2, 3, 7);
    EXPECT_ALL_EQUAL(mmult(A, B), B);

    auto D = RTensor::empty(2, 3, 7);
    mmult_into(D, A, B);
    EXPECT_ALL_EQUAL(D, B);
  }
}

TEST(RSparseMMultTest, DetectTensorDimensionMismatch) {
  EXPECT_THROW_DEBUG(mmult(RSparse::eye(2), RTensor{}), std::logic_error);
  EXPECT_THROW_DEBUG(mmult(RSparse::eye(2), RTensor::eye(3)), std::logic_error);
  EXPECT_THROW_DEBUG(mmult(RSparse::eye(2), RTensor::eye(3, 4, 5)),
                     std::logic_error);

  EXPECT_THROW_DEBUG(
      mmult_into(RTensor::empty(6), RSparse::eye(2), RTensor::random(2, 3)),
      std::logic_error);
  EXPECT_THROW_DEBUG(
      mmult_into(RTensor::empty(3, 3), RSparse::eye(2), RTensor::random(2, 3)),
      std::logic_error);
  EXPECT_THROW_DEBUG(
      mmult_into(RTensor::empty(2, 4), RSparse::eye(2), RTensor::random(2, 3)),
      std::logic_error);
  EXPECT_THROW_DEBUG(mmult_into(RTensor::empty(2, 3, 1), RSparse::eye(2),
                                RTensor::random(2, 3)),
                     std::logic_error);
}

TEST(RSparseMMultTest, ManyRandomMultiplications) {
  test_mmult<double, double>(MATRIX_MAX_DIM);
}

}  // namespace tensor_test
