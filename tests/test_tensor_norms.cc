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
class NormTest : public TensorTest<T> {};

using MyTypes = ::testing::Types<RTensor, CTensor>;
TYPED_TEST_SUITE(NormTest, MyTypes);

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
    ASSERT_FLOAT_EQ(norm2(P), sqrt(9.0 + 1.0 + 4.0));
  }
  {
    const auto z = this->small_number();
    const Tensor P = {4.0 * z, 3.0 * z};
    ASSERT_FLOAT_EQ(norm2(P), abs(z) * 5.0);
  }
}

TYPED_TEST(NormTest, Norm2IsAccurateInLargeVectors) {
  using Tensor = TypeParam;
  constexpr double sqrtpi2over6 = 1.28254983016186409;
  constexpr int steps = 100000000;
  constexpr double tolerance = 4e-7;
  constexpr double exact_norm2 = sqrtpi2over6;
  {
    SCOPED_TRACE("Larger to smaller");
    const Tensor P = linspace(this->one(), this->one() * double(steps), steps);
    const Tensor invP = this->one() / P;
    ASSERT_NEAR(norm2(invP), exact_norm2, tolerance);
  }
  {
    SCOPED_TRACE("Smaller to larger");
    const Tensor P = linspace(this->one() * double(steps), this->one(), steps);
    const Tensor invP = this->one() / P;
    ASSERT_NEAR(norm2(invP), exact_norm2, tolerance);
  }
}

}  // namespace
