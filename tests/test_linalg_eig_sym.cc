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
#include <tensor/tensor.h>
#include <tensor/linalg.h>

namespace tensor_test {

using namespace tensor;

//////////////////////////////////////////////////////////////////////
// EIGENVALUE DECOMPOSITIONS
//

template <typename elt_t>
void test_eye_eig_sym(int n) {
  if (n == 0) {
#ifdef TENSOR_DEBUG
    ASSERT_THROW_DEBUG(linalg::eig_sym(Tensor<elt_t>::eye(n, n)),
                       ::tensor::invalid_assertion);
#endif
    return;
  }
  Tensor<elt_t> Inn = Tensor<elt_t>::eye(n, n);
  Tensor<elt_t> U;
  RTensor s = linalg::eig_sym(Inn, &U);
  Tensor<elt_t> V = adjoint(U);
  EXPECT_TRUE(all_equal(Inn, U));
  EXPECT_TRUE(all_equal(Inn, V));
  EXPECT_TRUE(all_equal(s, RTensor::ones(n)));
}

template <typename elt_t>
void test_random_eig_sym(int n) {
  if (n > 0) {
    for (int times = 10; times; --times) {
      Tensor<elt_t> R, A = Tensor<elt_t>::random(n, n);
      A = mmult(A, adjoint(A)) / norm0(A);
      RTensor s = linalg::eig_sym(A, &R);
      Tensor<elt_t> L = adjoint(R);
      RTensor dS = diag(s);
      constexpr double epsilon = 1e-12;
      EXPECT_TRUE(norm0(abs(s) - s) < 1e-13);
      EXPECT_TRUE(unitaryp(R, 1e-10));
      EXPECT_CEQ3(mmult(A, R), mmult(R, dS), epsilon);
      EXPECT_CEQ3(mmult(L, A), mmult(dS, L), epsilon);
      EXPECT_CEQ3(A, mmult(R, mmult(dS, L)), epsilon);
    }
  }
}

//////////////////////////////////////////////////////////////////////
// GENERALIZED EIGENVALUE DECOMPOSITIONS
//

template <typename elt_t>
void test_eye_eig_sym_gen(int n) {
  if (n == 0) {
#ifdef TENSOR_DEBUG
    ASSERT_THROW_DEBUG(
        linalg::eig_sym(Tensor<elt_t>::eye(n, n), Tensor<elt_t>::eye(n, n)),
        ::tensor::invalid_assertion);
#endif
    return;
  }
  ASSERT_THROW_DEBUG(linalg::eig_sym(Tensor<elt_t>::eye(n, n),
                                     Tensor<elt_t>::eye(n + 1, n + 1)),
                     ::tensor::invalid_assertion);
  auto Inn = Tensor<elt_t>::eye(n, n);
  RTensor d = linspace(n, 1.0, n);
  auto Bnn = diag(d);
  Tensor<elt_t> U;
  RTensor s = linalg::eig_sym(Inn, Bnn, &U);
  EXPECT_ALL_NEAR(diag(elt_t(1.0) / sqrt(d)), U, EPSILON);
  EXPECT_ALL_NEAR(s, 1.0 / d, EPSILON);
}

template <typename elt_t>
void test_random_eig_sym_gen(int n) {
  if (n > 0) {
    for (int times = 10; times; --times) {
      Tensor<elt_t> A = Tensor<elt_t>::random(n, n), B = Tensor<elt_t>::eye(n),
                    R;
      A = mmult(A, adjoint(A)) / norm0(A);
      RTensor s = linalg::eig_sym(A, B, &R);
      Tensor<elt_t> L = adjoint(R);
      RTensor dS = diag(s);
      constexpr double epsilon = 1e-12;
      EXPECT_TRUE(norm0(abs(s) - s) < 1e-13);
      EXPECT_TRUE(unitaryp(R, 1e-10));
      EXPECT_ALL_NEAR(mmult(A, R), mmult(R, dS), epsilon);
      EXPECT_ALL_NEAR(mmult(L, A), mmult(dS, L), epsilon);
      EXPECT_ALL_NEAR(A, mmult(R, mmult(dS, L)), epsilon);
    }
  }
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

constexpr int max_tensor_size = 3;

TEST(RMatrixTest, EyeEigSymTest) {
  test_over_integers(0, max_tensor_size, test_eye_eig_sym<double>);
}

TEST(RMatrixTest, RandomEigSymTest) {
  test_over_integers(0, max_tensor_size, test_random_eig_sym<double>);
}

TEST(RMatrixTest, EyeEigSymGenTest) {
  test_over_integers(0, max_tensor_size, test_eye_eig_sym_gen<double>);
}

TEST(RMatrixTest, RandomEigSymGenTest) {
  test_over_integers(0, max_tensor_size, test_random_eig_sym_gen<double>);
}

//////////////////////////////////////////////////////////////////////
// COMPLEX SPECIALIZATIONS
//

TEST(CMatrixTest, EyeEigSymTest) {
  test_over_integers(0, max_tensor_size, test_eye_eig_sym<cdouble>);
}

TEST(CMatrixTest, RandomEigSymTest) {
  test_over_integers(0, max_tensor_size, test_random_eig_sym<cdouble>);
}

TEST(CMatrixTest, EyeEigSymGenTest) {
  test_over_integers(0, max_tensor_size, test_eye_eig_sym_gen<cdouble>);
}

TEST(CMatrixTest, RandomEigSymGenTest) {
  test_over_integers(0, max_tensor_size, test_random_eig_sym_gen<cdouble>);
}

}  // namespace tensor_test
