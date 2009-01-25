// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include "loops.h"
#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>
#include <tensor/tensor.h>

#include "slow_mmult.cc"

namespace tensor_test {

  //////////////////////////////////////////////////////////////////////
  // MATRIX MULTIPLICATION
  //

  template<typename n1, typename n2>
  void test_mmult(index max_dim) {
    index m = 0;
    for (index i = 1; i <= max_dim; i++) {
      for (index j = 1; j <= max_dim; j++) {
        for (index k = 1; k <= max_dim; k++) {
          Tensor<n1> A(i,j), B(j,k);
          A.randomize();
          B.randomize();
          EXPECT_TRUE(approx_eq(mmult(A, B), fold_22_12(A, B)));
          unique(A);
          unique(B);
          if (m % 10 == 0)
            std::cout << '.' << std::flush;
          m++;
          if (m % 600 == 0)
            std::cout << std::endl;
        }
      }
    }
    std::cout << std::endl;
    ASSERT_DEATH(mmult(Tensor<n1>::eye(1,0), Tensor<n2>::ones(0,3)), ".*");
  }

  //////////////////////////////////////////////////////////////////////
  // REAL SPECIALIZATIONS
  //

#define MATRIX_MAX_DIM 24

  TEST(MmultTest, MmultDoubleDoubleTest) {
    test_mmult<double,double>(MATRIX_MAX_DIM);
  }

  TEST(MmultTest, MmultCdoubleCdoubleTest) {
    test_mmult<cdouble,cdouble>(MATRIX_MAX_DIM);
  }

} // namespace tensor_test
