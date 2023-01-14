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

#include <tensor/tensor.h>
#include <tensor/sparse.h>
#include "loops.h"
#include <gtest/gtest.h>

namespace tensor_test {

using namespace tensor;
using tensor::index;

template <typename elt_t>
void test_sparse_binop_zeros(Tensor<elt_t> &t) {
  tensor::index rows = t.rows(), cols = t.columns();
  // Operations among empty sparses must work. They did not because
  // we did not special-case them and there was an error in the code.
  {
    Sparse<elt_t> A = Sparse<elt_t>(rows, cols);
    Sparse<elt_t> B = Sparse<elt_t>(rows, cols);
    EXPECT_TRUE(all_equal(full(A + B), full(A) + full(B)));
  }
  {
    Sparse<elt_t> A = Sparse<elt_t>(rows, cols);
    Sparse<elt_t> B = Sparse<elt_t>(rows, cols);
    EXPECT_TRUE(all_equal(full(A - B), full(A) - full(B)));
  }
  {
    Sparse<elt_t> A = Sparse<elt_t>(rows, cols);
    Sparse<elt_t> B = Sparse<elt_t>(rows, cols);
    EXPECT_TRUE(all_equal(full(A * B), full(A) * full(B)));
  }
}

TEST(RSparseTest, BinopZeros) {
  test_over_fixed_rank_tensors<double>(test_sparse_binop_zeros<double>, 2, 7);
}
TEST(CSparseTest, BinopZeros) {
  test_over_fixed_rank_tensors<cdouble>(test_sparse_binop_zeros<cdouble>, 2, 7);
}

template <typename elt_t>
void test_sparse_binop_random(Tensor<elt_t> &t) {
  tensor::index rows = t.rows(), cols = t.columns();
  for (int i = 0; i < rows * cols; i++) {
    {
      Sparse<elt_t> A = Sparse<elt_t>::random(rows, cols);
      Sparse<elt_t> B = Sparse<elt_t>::random(rows, cols);
      EXPECT_TRUE(all_equal(full(A + B), full(A) + full(B)));
    }
    {
      Sparse<elt_t> A = Sparse<elt_t>::random(rows, cols);
      Sparse<elt_t> B = Sparse<elt_t>::random(rows, cols);
      EXPECT_TRUE(all_equal(full(A - B), full(A) - full(B)));
    }
    {
      Sparse<elt_t> A = Sparse<elt_t>::random(rows, cols);
      Sparse<elt_t> B = Sparse<elt_t>::random(rows, cols);
      EXPECT_TRUE(all_equal(full(A * B), full(A) * full(B)));
    }
  }
}

TEST(RSparseTest, BinopSmall) {
  // Particular test cases that showed to have problems with the
  // old code.
  {
    RSparse A(RTensor::zeros(1, 1));
    RSparse B(Tensor2D<double>({{0.958326}}));
    EXPECT_TRUE(all_equal(full(A + B), full(A) + full(B)));
    EXPECT_TRUE(all_equal(full(A - B), full(A) - full(B)));
    EXPECT_TRUE(all_equal(full(A * B), full(A) * full(B)));
  }
  {
    RSparse A(Tensor2D<double>({{0.958326}, {0.0}}));
    RSparse B(RTensor::zeros(2, 1));
    EXPECT_TRUE(all_equal(full(A + B), full(A) + full(B)));
    EXPECT_TRUE(all_equal(full(A - B), full(A) - full(B)));
    EXPECT_TRUE(all_equal(full(A * B), full(A) * full(B)));
  }
}

TEST(RSparseTest, BinopRandom) {
  test_over_fixed_rank_tensors<double>(test_sparse_binop_random<double>, 2, 7);
}

TEST(CSparseTest, BinopRandom) {
  test_over_fixed_rank_tensors<cdouble>(test_sparse_binop_random<cdouble>, 2,
                                        7);
}

}  // namespace tensor_test
