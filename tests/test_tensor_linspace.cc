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

#include <cmath>
#include <tensor/tensor.h>
#include <tensor/tools.h>
#include "loops.h"

namespace tensor_test {

//
// Linspace with number arguments
//
#ifndef __cppcheck__

template <typename T>
class TestNumberLinspace : public TensorTest<T> {};

using MyTypes = ::testing::Types<RTensor, CTensor>;
TYPED_TEST_SUITE(TestNumberLinspace, MyTypes);

TYPED_TEST(TestNumberLinspace, SizeZeroGivesZeroSizeVector) {
  using Tensor = TypeParam;

  auto start = this->to_value_type(0);
  auto end = this->to_value_type(pi);
  ASSERT_EQ(linspace(start, end, 0).rank(), 1);
  ASSERT_EQ(linspace(start, end, 0).size(), 0);
  ASSERT_EQ(linspace(start, end, 0).dimension(0), 0);
}

TYPED_TEST(TestNumberLinspace, SizeOneGivesFirstValue) {
  using Tensor = TypeParam;

  ASSERT_TRUE(
      all_equal(linspace(this->to_value_type(13), this->to_value_type(10), 1),
                Tensor{this->to_value_type(13)}));
}

TYPED_TEST(TestNumberLinspace, OtherSizeGivesVectorOfThatSize) {
  using Tensor = TypeParam;

  auto start = this->to_value_type(pi);
  auto end = this->to_value_type(-pi);

  ASSERT_EQ(linspace(start, end, 0).size(), 0);
  ASSERT_EQ(linspace(start, end, 1).size(), 1);
  ASSERT_EQ(linspace(start, end, 2).size(), 2);
  ASSERT_EQ(linspace(start, end, 10).size(), 10);

  ASSERT_EQ(linspace(start, end, 0).rank(), 1);
  ASSERT_EQ(linspace(start, end, 1).rank(), 1);
  ASSERT_EQ(linspace(start, end, 2).rank(), 1);
  ASSERT_EQ(linspace(start, end, 10).rank(), 1);
}

TYPED_TEST(TestNumberLinspace, SizeTwoGivesIntervalBoundaries) {
  using Tensor = TypeParam;

  auto start = this->to_value_type(pi);
  auto end = this->to_value_type(-pi);

  ASSERT_TRUE(all_equal(linspace(start, end, 2), Tensor{start, end}));
}

TYPED_TEST(TestNumberLinspace, ProducesEquispacedVectors) {
  using Tensor = TypeParam;

  auto P = linspace(this->to_value_type(0.0), this->to_value_type(pi), 4);
  auto delta = P[1] - P[0];
  ASSERT_CEQ(pi / 3.0, delta);
  ASSERT_CEQ(delta, P[2] - P[1]);
  ASSERT_CEQ(delta, P[3] - P[2]);
}

TYPED_TEST(TestNumberLinspace, WithCommensurateSizeProducesIntegerVectors) {
  using Tensor = TypeParam;

  auto size = 4;
  auto P =
      linspace(this->to_value_type(0), this->to_value_type(size), size + 1);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(P[i], this->to_value_type(i));
  }
}

//
// Linspace with number arguments
//

template <typename T>
class TestTensorLinspace : public TensorTest<T> {};

using MyTypes = ::testing::Types<RTensor, CTensor>;
TYPED_TEST_SUITE(TestTensorLinspace, MyTypes);

TYPED_TEST(TestTensorLinspace, SizeAddsDimensionOfThatSize) {
  using Tensor = TypeParam;

  {
    auto start = Tensor::random(4);
    auto end = Tensor::random(4);

    ASSERT_TRUE(
        all_equal(linspace(start, end, 0).dimensions(), Dimensions{4, 0}));
    ASSERT_TRUE(
        all_equal(linspace(start, end, 1).dimensions(), Dimensions{4, 1}));
    ASSERT_TRUE(
        all_equal(linspace(start, end, 2).dimensions(), Dimensions{4, 2}));
    ASSERT_TRUE(
        all_equal(linspace(start, end, 10).dimensions(), Dimensions{4, 10}));
  }
  {
    auto start = Tensor::random(2, 3);
    auto end = Tensor::random(2, 3);

    ASSERT_TRUE(
        all_equal(linspace(start, end, 0).dimensions(), Dimensions{2, 3, 0}));
    ASSERT_TRUE(
        all_equal(linspace(start, end, 1).dimensions(), Dimensions{2, 3, 1}));
    ASSERT_TRUE(
        all_equal(linspace(start, end, 2).dimensions(), Dimensions{2, 3, 2}));
    ASSERT_TRUE(
        all_equal(linspace(start, end, 10).dimensions(), Dimensions{2, 3, 10}));
  }
}

TYPED_TEST(TestTensorLinspace, SizeOneGivesFirstValue) {
  using Tensor = TypeParam;
  {
    auto start = Tensor{-1.0, 1.0};
    auto end = Tensor{pi, -pi};
    auto tensor = linspace(start, end, 1);
    ASSERT_CEQ(tensor(0, 0), start[0]);
    ASSERT_CEQ(tensor(1, 0), start[1]);
  }
  {
    auto start = Tensor::random(2, 2);
    auto end = Tensor::random(2, 2);
    auto tensor = linspace(start, end, 1);
    ASSERT_CEQ(tensor(0, 0, 0), start(0, 0));
    ASSERT_CEQ(tensor(0, 1, 0), start(0, 1));
    ASSERT_CEQ(tensor(1, 0, 0), start(1, 0));
    ASSERT_CEQ(tensor(1, 1, 0), start(1, 1));
  }
}

TYPED_TEST(TestTensorLinspace, SizeTwoGivesIntervalBoundaries) {
  using Tensor = TypeParam;

  auto start = Tensor::random(2, 3);
  auto end = Tensor::random(2, 3);
  auto tensor = linspace(start, end, 2);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      ASSERT_CEQ(tensor(i, j, 0), start(i, j));
      ASSERT_CEQ(tensor(i, j, 1), end(i, j));
    }
  }
}

TYPED_TEST(TestTensorLinspace, ProducesEquispacedVectors) {
  using Tensor = TypeParam;

  auto start = Tensor::random(2, 3);
  auto end = Tensor::random(2, 3);
  auto tensor = linspace(start, end, 4);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      auto delta = tensor(i, j, 1) - tensor(i, j, 0);
      ASSERT_CEQ((end(i, j) - start(i, j)) / 3.0, delta);
      ASSERT_CEQ(delta, tensor(i, j, 2) - tensor(i, j, 1));
      ASSERT_CEQ(delta, tensor(i, j, 3) - tensor(i, j, 2));
    }
  }
}

TYPED_TEST(TestTensorLinspace, WithCommensurateSizeProducesIntegerVectors) {
  using Tensor = TypeParam;

  auto size = 4;
  auto start = Tensor{0, 3};
  auto end = Tensor{size + 0.0, size + 3.0};
  auto tensor = linspace(start, end, size + 1);

  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(tensor(0, i), this->to_value_type(i));
    ASSERT_EQ(tensor(1, i), this->to_value_type(i + 3));
  }
}

#endif // __cppcheck__

}  // namespace tensor_test
