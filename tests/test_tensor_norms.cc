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

#include <gtest/gtest.h>
#include <cmath>
#include <tensor/tensor.h>

namespace {

using namespace ::tensor;

template <typename T>
class NormTest : public ::testing::Test {
 public:
  using value_type = typename T::elt_t;

  value_type small_number() const { return 1e-6 * rand<value_type>(); }

  constexpr value_type one() const { return number_one<value_type>(); }
};

using MyTypes = ::testing::Types<RTensor, CTensor>;
TYPED_TEST_SUITE(NormTest, MyTypes);

static constexpr double pi = 3.14159265358979323846;
static constexpr double pi2over6 = 1.64493406684822643647241;

//
// NORM-0
//

TYPED_TEST(NormTest, Norm0IsZeroForEmptyTensors) {
  using Tensor = TypeParam;

  ASSERT_EQ(norm0(Tensor()), 0);
}

TYPED_TEST(NormTest, Norm0IsNonNegative) {
  using Tensor = TypeParam;

  const Tensor P = {0.0, -3.0, 1.0, -2.0};
  ASSERT_TRUE(norm0(P) >= 0);
}

TYPED_TEST(NormTest, Norm0ReturnsLargestAbsoluteValue) {
  using Tensor = TypeParam;

  const Tensor P = {0.0, -3.0, 1.0, -2.0};
  ASSERT_EQ(norm0(P), 3.0);
  const Tensor P2 = P + this->small_number();
  ASSERT_FLOAT_EQ(norm0(P), abs(P2[1]));
}

//
// NORM-2
//

TYPED_TEST(NormTest, Norm2IsZeroForEmptyTensors) {
  using Tensor = TypeParam;

  ASSERT_EQ(norm2(Tensor()), 0);
}

TYPED_TEST(NormTest, Norm2IsNonNegative) {
  using Tensor = TypeParam;

  const Tensor P = {0.0, -3.0, 1.0, -2.0};
  ASSERT_TRUE(norm2(P) >= 0);
}

TYPED_TEST(NormTest, Norm2ReturnsSqrtOfSumOfSquares) {
  using Tensor = TypeParam;
  {
    const Tensor P = {0.0, -3.0, 1.0, -2.0};
    ASSERT_EQ(norm2(P), sqrt(9.0 + 1.0 + 4.0));
  }
  {
    const auto z = this->small_number();
    const Tensor P = {4.0 * z, 3.0 * z};
    ASSERT_FLOAT_EQ(norm2(P), abs(z) * 5.0);
  }
}

}  // namespace
