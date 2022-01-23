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

#include <algorithm>
#include <functional>
#include "loops.h"
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include <tensor/linalg.h>

namespace tensor_test {

using namespace tensor;

template <typename elt_t>
Tensor<elt_t> random_svd_matrix(int n, int m, RTensor &s) {
  if (n == 0 || m == 0) {
    return Tensor<elt_t>::zeros(n, m);
  } else {
    Tensor<elt_t> U = random_unitary<elt_t>(n);
    Tensor<elt_t> V = random_unitary<elt_t>(m);
    EXPECT_TRUE(unitaryp(U));
    EXPECT_TRUE(unitaryp(V));
    s = RTensor(igen << std::min(n, m));
    s.randomize();
    s = abs(s);  // Just in case we change our mind and make rand < 0
    return mmult(U, mmult(diag(s, 0, n, m), V));
  }
}

//////////////////////////////////////////////////////////////////////
// SINGULAR VALUE DECOMPOSITIONS
//

template <typename elt_t, bool block>
void test_eye_svd(int n) {
  if (n == 0) {
#ifndef NDEBUG
    ASSERT_DEATH(linalg::svd(Tensor<elt_t>::eye(n, n)), ".*");
#endif
    return;
  }
  for (int m = 1; m < n; m++) {
    Tensor<elt_t> Imn = Tensor<elt_t>::eye(m, n);
    Tensor<elt_t> Imm = Tensor<elt_t>::eye(m, m);
    Tensor<elt_t> Inn = Tensor<elt_t>::eye(n, n);
    RTensor s1 = RTensor::ones(Dimensions({std::min(m, n)}));
    Tensor<elt_t> U, V;
    RTensor s;

    s = block ? linalg::block_svd(Imn, &U, &V, false)
              : linalg::svd(Imn, &U, &V, false);
    EXPECT_TRUE(all_equal(Imm, U));
    EXPECT_TRUE(all_equal(Inn, V));
    EXPECT_TRUE(all_equal(s, s1));
    s = block ? linalg::block_svd(Imn, &U, &V, true)
              : linalg::svd(Imn, &U, &V, true);
    EXPECT_TRUE(all_equal(U, m < n ? Imm : Imn));
    EXPECT_TRUE(all_equal(V, m < n ? Imn : Inn));
    EXPECT_TRUE(all_equal(s, s1));
  }
}

template <typename elt_t, bool block>
void test_random_svd(int n) {
  if (n == 0) {
#ifndef NDEBUG
    ASSERT_DEATH(linalg::svd(Tensor<elt_t>::eye(n, n)), ".*");
#endif
    return;
  }
  for (int times = 0; times < 1; ++times) {
    for (int m = 1; m < 2 * n; ++m) {
#if 0
        Tensor<elt_t> A(m,n);
        A.randomize();
#else
      RTensor true_s;
      Tensor<elt_t> A = random_svd_matrix<elt_t>(m, n, true_s);
      std::sort(true_s.begin(), true_s.end(), std::greater<double>());
#endif

      Tensor<elt_t> U, Vt;
      RTensor s = block ? linalg::block_svd(A, &U, &Vt, false)
                        : linalg::svd(A, &U, &Vt, false);
      EXPECT_TRUE(unitaryp(U, 1e-10));
      EXPECT_TRUE(unitaryp(Vt, 1e-10));
      EXPECT_TRUE(approx_eq(abs(s), s));
      EXPECT_TRUE(approx_eq(A, mmult(U, mmult(diag(s, 0, m, n), Vt))));

      EXPECT_TRUE(approx_eq(true_s, s));

      RTensor s2 = block ? linalg::block_svd(A, &U, &Vt, true)
                         : linalg::svd(A, &U, &Vt, true);
      EXPECT_TRUE(approx_eq(s, s2));
      EXPECT_TRUE(unitaryp(U));
      EXPECT_TRUE(unitaryp(Vt));
      EXPECT_TRUE(approx_eq(A, mmult(U, mmult(diag(s), Vt))));
    }
  }
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

TEST(RMatrixTest, EyeSvdTest) {
  test_over_integers(0, 32, test_eye_svd<double, false>);
}

TEST(RMatrixTest, RandomSvdTest) {
  test_over_integers(0, 32, test_random_svd<double, false>);
}

TEST(RMatrixTest, EyeBlockSvdTest) {
  test_over_integers(0, 32, test_eye_svd<double, true>);
}

TEST(RMatrixTest, RandomSingleBlockSvdTest) {
  test_over_integers(0, 32, test_random_svd<double, true>);
}

//////////////////////////////////////////////////////////////////////
// COMPLEX SPECIALIZATIONS
//

TEST(CMatrixTest, EyeSvdTest) {
  test_over_integers(0, 32, test_eye_svd<cdouble, false>);
}

TEST(CMatrixTest, RandomSvdTest) {
  test_over_integers(0, 32, test_random_svd<cdouble, false>);
}

TEST(CMatrixTest, EyeBlockSvdTest) {
  test_over_integers(0, 32, test_eye_svd<cdouble, true>);
}

TEST(CMatrixTest, RandomSingleBlockSvdTest) {
  test_over_integers(0, 32, test_random_svd<cdouble, true>);
}

}  // namespace tensor_test
