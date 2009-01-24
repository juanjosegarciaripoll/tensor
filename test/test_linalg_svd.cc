// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include "loops.h"
#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>
#include <tensor/tensor.h>
#include <tensor/linalg.h>

namespace tensor_test {

  using namespace tensor;

  //////////////////////////////////////////////////////////////////////
  // SINGULAR VALUE DECOMPOSITIONS
  //

  template<typename elt_t>
  bool unitaryp(const Tensor<elt_t> &U)
  {
    Tensor<elt_t> Ut = adjoint(U);
    return true;
  }

  template<typename elt_t>
  void test_eye_svd(int n) {
    if (n == 0) {
#ifndef NDEBUG
      ASSERT_DEATH(linalg::svd(Tensor<elt_t>::eye(n,n)), ".*");
#endif
      return;
    }
    for (int m = 1; m < n; m++) {
      Tensor<elt_t> Imn = Tensor<elt_t>::eye(m,n);
      Tensor<elt_t> Imm = Tensor<elt_t>::eye(m,m);
      Tensor<elt_t> Inn = Tensor<elt_t>::eye(n,n);
      Tensor<elt_t> V1(std::min(m,n));
      V1.fill_with(number_one<elt_t>());
      Tensor<elt_t> U, V;
      RTensor s;
      
      s = linalg::svd(Imn, &U, &V, false);
      EXPECT_EQ(Imm, U);
      EXPECT_EQ(Inn, V);
      EXPECT_EQ(s, V1);
      s = linalg::svd(Imn, &U, &V, true);
      EXPECT_EQ(U, m<n? Imm : Imn);
      EXPECT_EQ(V, m<n? Imn : Inn);
      EXPECT_EQ(s, V1);
    }
  }

  template<typename elt_t>
  void test_random_svd(int n) {
    if (n == 0) {
#ifndef NDEBUG
      ASSERT_DEATH(linalg::svd(Tensor<elt_t>::eye(n,n)), ".*");
#endif
      return;
    }
    for (int times = 10; times; --times) {
      for (int m = 1; m < 2*n; ++m) {
        Tensor<elt_t> A(m,n);
        A.randomize();

        Tensor<elt_t> U, V;
        RTensor s = linalg::svd(A, &U, &V, false);

        EXPECT_TRUE(unitaryp(U));
        EXPECT_TRUE(unitaryp(V));
        EXPECT_EQ(abs(s), s);

        RTensor s2 = linalg::svd(A, &U, &V, true);
        EXPECT_EQ(s, s2);
        EXPECT_TRUE(unitaryp(U));
        EXPECT_TRUE(unitaryp(V));
      }
    }
  }

  //////////////////////////////////////////////////////////////////////
  // REAL SPECIALIZATIONS
  //
  
  TEST(RMatrixTest, EyeSvdTest) {
    test_over_integers(0, 32, test_eye_svd<double>);
  }

  TEST(RMatrixTest, RandomSvdTest) {
    test_over_integers(0, 32, test_random_svd<double>);
  }

  //////////////////////////////////////////////////////////////////////
  // COMPLEX SPECIALIZATIONS
  //

  TEST(CMatrixTest, EyeSvdTest) {
    test_over_integers(0, 32, test_eye_svd<cdouble>);
  }

  TEST(CMatrixTest, RandomSvdTest) {
    test_over_integers(0, 32, test_random_svd<cdouble>);
  }

} // namespace linalg_test
