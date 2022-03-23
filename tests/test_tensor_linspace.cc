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

}  // namespace tensor_test
