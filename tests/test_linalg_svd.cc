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
#include "../src/linalg/find_blocks.hpp"

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
    s = abs(RTensor::random(
        std::min(n, m)));  // Just in case we change our mind and make rand < 0
    return mmult(U, mmult(diag(s, 0, n, m), V));
  }
}

//////////////////////////////////////////////////////////////////////
// SINGULAR VALUE DECOMPOSITIONS
//
template <typename Tensor>
void test_svd(const Tensor &A, const RTensor &exact_s, const Tensor &exact_U,
              const Tensor &exact_Vt, bool block, bool economic) {
  Tensor U, Vt;
  RTensor s;
  const char *scopes[2][2] = {
      {"normal svd, full matrices", "normal svd, economic"},
      {"block svd, full matrices", "block svd, economic"}};
  SCOPED_TRACE(scopes[block][economic]);
  s = block ? linalg::block_svd(A, &U, &Vt, economic)
            : linalg::svd(A, &U, &Vt, economic);
  ASSERT_CEQ(s, exact_s);
  auto reconstructed_A =
      mmult(U, mmult(diag(s, 0, U.columns(), Vt.rows()), Vt));
  ASSERT_CEQ(A, reconstructed_A);
  ASSERT_CEQ(abs(U), abs(exact_U));
  ASSERT_CEQ(abs(Vt), abs(exact_Vt));
}

template <typename elt_t, bool block>
void test_eye_svd(int n) {
  if (n == 0) {
#ifdef TENSOR_DEBUG
    ASSERT_THROW_DEBUG(linalg::svd(Tensor<elt_t>::eye(n, n)),
                 ::tensor::invalid_assertion);
#endif
    return;
  }
  for (int m = 1; m < n; m++) {
    Tensor<elt_t> A = Tensor<elt_t>::eye(m, n);
    RTensor exact_s = RTensor::ones(std::min(m, n));
    {
      auto exact_U = Tensor<elt_t>::eye(m, m);
      auto exact_V = Tensor<elt_t>::eye(n, n);
      bool economic = false;
      test_svd(A, exact_s, exact_U, exact_V, block, economic);
    }
    {
      auto exact_U = Tensor<elt_t>::eye(m, exact_s.size());
      auto exact_V = Tensor<elt_t>::eye(exact_s.size(), n);
      bool economic = true;
      test_svd(A, exact_s, exact_U, exact_V, block, economic);
    }
  }
}

template <typename elt_t, bool block>
void test_random_svd(int n) {
  if (n == 0) {
#ifdef TENSOR_DEBUG
    ASSERT_THROW_DEBUG(linalg::svd(Tensor<elt_t>::eye(n, n)),
                 ::tensor::invalid_assertion);
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
      auto A2 = A;

      RTensor s2 = block ? linalg::block_svd(A, &U, &Vt, true)
                         : linalg::svd(A, &U, &Vt, true);
      if (!approx_eq(s, s2)) {
        std::vector<Indices> rows, cols;
        std::cerr << "find_blocks=" << linalg::find_blocks(A, rows, cols, 0.0)
                  << '\n';
        std::cerr << "block=" << block << '\n';
        std::cerr << "s=" << s << '\n' << "s2=" << s2 << '\n';
        std::cerr << "A=" << A << '\n' << "A2=" << A2 << '\n';
      }
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

TEST(RMatrixTest, RowTest) {
  bool block, economic;
  double z = 1 / sqrt(2.0);
  RTensor A{{z, z, 0.0}};
  RTensor exact_s{1.0};
  RTensor exact_U = RTensor::eye(1);
  RTensor exact_V{{z, z, 0.0}};
  test_svd(A, exact_s, exact_U, exact_V, block = false, economic = true);
  test_svd(A, exact_s, exact_U, exact_V, block = true, economic = true);
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
