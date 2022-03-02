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
#include <tensor/tensor.h>
#include "loops.h"

namespace tensor_test {

using namespace tensor;
using tensor::index;

template <typename elt_t>
elt_t slow_trace(const Tensor<elt_t> &A) {
  elt_t output = number_zero<elt_t>();
  for (index i = 0; i < std::min(A.rows(), A.columns()); i++)
    output = output + A(i, i);
  return output;
}

template <typename elt_t>
void test_matrix_trace() {
  for (index rows = 1; rows <= 4; rows++) {
    for (index cols = 1; cols <= 4; cols++) {
      Tensor<elt_t> A = Tensor<elt_t>::random(rows, cols);
      elt_t t = slow_trace(A);
      Tensor<elt_t> T{gen(t)};
      EXPECT_EQ(trace(A), t);
      EXPECT_TRUE(all_equal(trace(A, 0, -1), T));
      EXPECT_TRUE(all_equal(trace(reshape(A, 1, rows, cols), 1, 2), T));
      EXPECT_TRUE(all_equal(trace(reshape(A, 1, rows, 1, cols), 1, 3),
                            reshape(T, 1, 1)));
      EXPECT_TRUE(all_equal(trace(reshape(A, 1, rows, 1, cols, 1), 1, 3),
                            reshape(T, 1, 1, 1)));
      EXPECT_TRUE(all_equal(trace(reshape(A, 1, rows, cols, 1), 1, 2),
                            reshape(T, 1, 1)));
      EXPECT_TRUE(all_equal(trace(reshape(A, rows, 1, cols, 1), 0, 2),
                            reshape(T, 1, 1)));
      EXPECT_TRUE(all_equal(trace(reshape(A, rows, 1, cols), 0, 2), T));
    }
  }
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

TEST(TensorTrace, RMatrix) { test_matrix_trace<double>(); }

#if defined(GTEST_HAS_DEATH_TEST) && !defined(NDEBUG)
TEST(TensorTrace, MatrixTraceExpectsRMatrix) {
  ASSERT_DEATH(trace(RTensor::zeros(1, 1, 1)), ".*");
}
#endif

//////////////////////////////////////////////////////////////////////
// COMPLEX SPECIALIZATIONS
//

TEST(TensorTrace, CMatrix) { test_matrix_trace<cdouble>(); }

#if defined(GTEST_HAS_DEATH_TEST) && !defined(NDEBUG)
TEST(TensorTrace, MatrixTraceExpectsCMatrix) {
  ASSERT_DEATH(trace(CTensor::zeros(1, 1, 1)), ".*");
}
#endif

}  // namespace tensor_test
