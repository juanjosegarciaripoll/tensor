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

#include "loops.h"
#include <gtest/gtest.h>
#include <tensor/tensor.h>

namespace tensor_test {

//////////////////////////////////////////////////////////////////////
// MATRIX CONSTRUCTORS
//

template <typename elt_t>
void test_ones(int n) {
  elt_t one = number_one<elt_t>();
  SCOPED_TRACE("vector");
  {
    Tensor<elt_t> M = Tensor<elt_t>::ones(n);
    EXPECT_EQ(1, M.rank());
    EXPECT_EQ(n, M.dimension(0));
    size_t ones = std::count(M.begin_const(), M.end_const(), one);
    EXPECT_EQ(M.size(), ones);
    EXPECT_EQ(1, M.ref_count());
  }
  SCOPED_TRACE("rectangular matrix");
  for (int i = 0; i <= n; i++) {
    {
      Tensor<elt_t> M = Tensor<elt_t>::ones(i, n);
      EXPECT_EQ(2, M.rank());
      EXPECT_EQ(i, M.rows());
      EXPECT_EQ(n, M.columns());
      size_t ones = std::count(M.begin_const(), M.end_const(), one);
      EXPECT_EQ(M.size(), ones);
      EXPECT_EQ(1, M.ref_count());
    }
    {
      Tensor<elt_t> M = Tensor<elt_t>::ones(n, i);
      EXPECT_EQ(2, M.rank());
      EXPECT_EQ(n, M.rows());
      EXPECT_EQ(i, M.columns());
      size_t ones = std::count(M.begin_const(), M.end_const(), one);
      EXPECT_EQ(M.size(), ones);
      EXPECT_EQ(1, M.ref_count());
    }
  }
}

template <typename elt_t>
void test_zeros(int n) {
  elt_t zero = number_zero<elt_t>();
  SCOPED_TRACE("vector");
  {
    Tensor<elt_t> M = Tensor<elt_t>::zeros(n);
    EXPECT_EQ(1, M.rank());
    EXPECT_EQ(n, M.dimension(0));
    size_t zeros = std::count(M.begin_const(), M.end_const(), zero);
    EXPECT_EQ(M.size(), zeros);
    EXPECT_EQ(1, M.ref_count());
  }
  SCOPED_TRACE("rectangular matrix");
  for (int i = 0; i <= n; i++) {
    {
      Tensor<elt_t> M = Tensor<elt_t>::zeros(i, n);
      EXPECT_EQ(2, M.rank());
      EXPECT_EQ(i, M.rows());
      EXPECT_EQ(n, M.columns());
      size_t zeros = std::count(M.begin_const(), M.end_const(), zero);
      EXPECT_EQ(M.size(), zeros);
      EXPECT_EQ(1, M.ref_count());
    }
    {
      Tensor<elt_t> M = Tensor<elt_t>::zeros(n, i);
      EXPECT_EQ(2, M.rank());
      EXPECT_EQ(n, M.rows());
      EXPECT_EQ(i, M.columns());
      size_t zeros = std::count(M.begin_const(), M.end_const(), zero);
      EXPECT_EQ(M.size(), zeros);
      EXPECT_EQ(1, M.ref_count());
    }
  }
}

template <typename elt_t>
void test_diag(int n) {
  elt_t zero = number_zero<elt_t>();
  Tensor<elt_t> d(n);
  int i = 1;
  for (typename Tensor<elt_t>::iterator it = d.begin(); it != d.end(); it++) {
    *it = (i++);
  }
  SCOPED_TRACE("no rows / columns specified");
  for (i = -n; i <= n; i++) {
    Tensor<elt_t> M = diag(d, i);
    EXPECT_EQ(2, M.rank());
    if (i == 0) {
      EXPECT_EQ(n, M.rows());
      EXPECT_EQ(n, M.columns());
      if (n) {
        EXPECT_EQ(d(0), M(0, 0));
        EXPECT_EQ(d(n - 1), M(n - 1, n - 1));
      }
    } else if (i > 0) {
      if (n) {
        EXPECT_EQ(d(0), M(0, i));
        EXPECT_EQ(d(n - 1), M(n - 1, i + n - 1));
      }
    } else {
      if (n) {
        EXPECT_EQ(d(0), M(-i, 0));
        EXPECT_EQ(d(n - 1), M(-i + n - 1, n - 1));
      }
    }
  }
  SCOPED_TRACE("errors");
#ifndef NDEBUG
  ASSERT_DEATH(diag(d, -n - 1, n, n), ".*");
  ASSERT_DEATH(diag(d, n + 1, n, n), ".*");
#endif
}

template <typename elt_t>
void test_transpose(int n) {
  for (int m = 0; m <= n; m++) {
    Tensor<elt_t> A(n, m);
    A.randomize();

    Tensor<elt_t> At = transpose(A);
    EXPECT_EQ(At.rank(), 2);
    EXPECT_EQ(At.rows(), m);
    EXPECT_EQ(At.columns(), n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        EXPECT_TRUE(A(i, j) == At(j, i));
      }
    }

    Tensor<elt_t> Att = transpose(At);
    EXPECT_EQ(Att.rank(), 2);
    EXPECT_EQ(Att.rows(), n);
    EXPECT_EQ(Att.columns(), m);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        EXPECT_TRUE(A(i, j) == Att(i, j));
      }
    }
  }
}

template <typename elt_t>
void test_adjoint(int n) {
  for (int m = 0; m <= n; m++) {
    Tensor<elt_t> A(n, m);
    A.randomize();

    Tensor<elt_t> At = adjoint(A);
    EXPECT_EQ(At.rank(), 2);
    EXPECT_EQ(At.rows(), m);
    EXPECT_EQ(At.columns(), n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        EXPECT_TRUE(A(i, j) == conj(At(j, i)));
      }
    }

    Tensor<elt_t> Att = adjoint(At);
    EXPECT_EQ(Att.rank(), 2);
    EXPECT_EQ(Att.rows(), n);
    EXPECT_EQ(Att.columns(), m);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        EXPECT_TRUE(A(i, j) == Att(i, j));
      }
    }
  }
}

template <typename elt_t>
void test_permute(int n) {
  for (int m = 0; m <= n; m++) {
    Tensor<elt_t> A(n, m);
    A.randomize();

    Tensor<elt_t> At = transpose(A);
    Tensor<elt_t> Ap = permute(A);
    EXPECT_EQ(Ap.rank(), 2);
    EXPECT_EQ(Ap.rows(), m);
    EXPECT_EQ(Ap.columns(), n);
    for (typename Tensor<elt_t>::iterator it = At.begin(), ip = Ap.begin();
         ip != Ap.end(); ip++, it++) {
      EXPECT_TRUE(*ip == *it);
    }

    Tensor<elt_t> Att = permute(At);
    EXPECT_EQ(Att.rank(), 2);
    EXPECT_EQ(Att.rows(), n);
    EXPECT_EQ(Att.columns(), m);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        EXPECT_TRUE(A(i, j) == Att(i, j));
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

TEST(RMatrixTest, OnesTest) { test_over_integers(0, 10, test_ones<double>); }

TEST(RMatrixTest, ZerosTest) { test_over_integers(0, 10, test_zeros<double>); }

TEST(RMatrixTest, DiagTest) { test_over_integers(0, 10, test_diag<double>); }

TEST(RMatrixTest, TransposeTest) {
  test_over_integers(0, 10, test_transpose<double>);
}

TEST(RMatrixTest, PermuteTest) {
  test_over_integers(0, 10, test_permute<double>);
}

TEST(RMatrixTest, AdjointTest) {
  test_over_integers(0, 10, test_adjoint<double>);
}

//////////////////////////////////////////////////////////////////////
// COMPLEX SPECIALIZATIONS
//

TEST(CMatrixTest, OnesTest) { test_over_integers(0, 10, test_ones<cdouble>); }

TEST(CMatrixTest, ZerosTest) { test_over_integers(0, 10, test_zeros<cdouble>); }

TEST(CMatrixTest, TransposeTest) {
  test_over_integers(0, 10, test_transpose<cdouble>);
}

TEST(CMatrixTest, PermuteTest) {
  test_over_integers(0, 10, test_permute<cdouble>);
}

TEST(CMatrixTest, AdjointTest) {
  test_over_integers(0, 10, test_adjoint<cdouble>);
}

}  // namespace tensor_test
