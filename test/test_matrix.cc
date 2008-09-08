// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include "loops.h"
#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>
#include <tensor/tensor.h>

namespace tensor_test {

//////////////////////////////////////////////////////////////////////
// MATRIX CONSTRUCTORS
//

template<typename elt_t>
void test_ones(int n) {
  elt_t one = number_one<elt_t>();
  SCOPED_TRACE("square matrix");
  {
    Tensor<elt_t> M = Tensor<elt_t>::ones(n);
    EXPECT_EQ(2, M.rank());
    EXPECT_EQ(n, M.rows());
    EXPECT_EQ(n, M.columns());
    int ones = std::count(M.begin_const(), M.end_const(), one);
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
      int ones = std::count(M.begin_const(), M.end_const(), one);
      EXPECT_EQ(M.size(), ones);
      EXPECT_EQ(1, M.ref_count());
    }
    {
      Tensor<elt_t> M = Tensor<elt_t>::ones(n, i);
      EXPECT_EQ(2, M.rank());
      EXPECT_EQ(n, M.rows());
      EXPECT_EQ(i, M.columns());
      int ones = std::count(M.begin_const(), M.end_const(), one);
      EXPECT_EQ(M.size(), ones);
      EXPECT_EQ(1, M.ref_count());
    }
  }
}

template<typename elt_t>
void test_zeros(int n) {
  elt_t zero = number_zero<elt_t>();
  SCOPED_TRACE("square matrix");
  {
    Tensor<elt_t> M = Tensor<elt_t>::zeros(n);
    EXPECT_EQ(2, M.rank());
    EXPECT_EQ(n, M.rows());
    EXPECT_EQ(n, M.columns());
    int zeros = std::count(M.begin_const(), M.end_const(), zero);
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
      int zeros = std::count(M.begin_const(), M.end_const(), zero);
      EXPECT_EQ(M.size(), zeros);
      EXPECT_EQ(1, M.ref_count());
    }
    {
      Tensor<elt_t> M = Tensor<elt_t>::zeros(n, i);
      EXPECT_EQ(2, M.rank());
      EXPECT_EQ(n, M.rows());
      EXPECT_EQ(i, M.columns());
      int zeros = std::count(M.begin_const(), M.end_const(), zero);
      EXPECT_EQ(M.size(), zeros);
      EXPECT_EQ(1, M.ref_count());
    }
  }
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

TEST(RMatrixTest, OnesTest) {
  test_over_integers(0, 10, test_ones<double>);
}

TEST(RMatrixTest, ZerosTest) {
  test_over_integers(0, 10, test_zeros<double>);
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

TEST(CMatrixTest, OnesTest) {
  test_over_integers(0, 10, test_ones<cdouble>);
}

TEST(CMatrixTest, ZerosTest) {
  test_over_integers(0, 10, test_zeros<cdouble>);
}

} // namespace tensor_test
