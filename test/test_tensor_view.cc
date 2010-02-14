// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include "loops.h"
#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>
#include <tensor/tensor.h>

namespace tensor_test {

  using namespace tensor;

  template<typename elt_t> void test_range1(Tensor<elt_t> &P) {
    Tensor<elt_t> Paux = P;

    for (tensor::index i = 0; i < P.dimension(0); i++) {
      Tensor<elt_t> t = P(range(i));
      EXPECT_EQ(P.rank(), t.rank());
      EXPECT_EQ(1, t.size());
      EXPECT_EQ(t[0], P(i));
      unchanged(P, Paux);

      Tensor<elt_t> t3 = P(range(i,i));
      EXPECT_EQ(t3, t);
      unchanged(P, Paux);

      Tensor<elt_t> t4 = P(range(i,i,1));
      EXPECT_EQ(t4, t);
      unchanged(P, Paux);

      if (i+1 < P.dimension(0)) {
        Tensor<elt_t> t5 = P(range(i,i+1,2));
        EXPECT_EQ(t5, t);
        unchanged(P, Paux);
      }

      Indices ndx(1);
      ndx.at(0) = i;
      Tensor<elt_t> t6 = P(range(ndx));
      EXPECT_EQ(t6, t);
      unchanged(P, Paux);
    }
    Tensor<elt_t> t = P(range());
    EXPECT_EQ(P, t);
    unchanged(P, Paux);
  }

  template<typename elt_t> void test_range2(Tensor<elt_t> &P) {
    Tensor<elt_t> Paux = P;

    for (tensor::index i = 0; i < P.dimension(0); i++) {
      for (tensor::index j = 0; j < P.dimension(1); j++) {
        Tensor<elt_t> t = P(range(i), range(j));
        EXPECT_EQ(P.rank(), t.rank());
        EXPECT_EQ(1, t.size());
        EXPECT_EQ(t[0], P(i,j));
        unchanged(P, Paux);

        Tensor<elt_t> t2 = P(range(i), range(j,j));
        EXPECT_EQ(t2, t);
        unchanged(P, Paux);

        Tensor<elt_t> t3 = P(range(i,i), range(j,j));
        EXPECT_EQ(t3, t);
        unchanged(P, Paux);

        Tensor<elt_t> t4 = P(range(i,i), range(j));
        EXPECT_EQ(t4, t);
        unchanged(P, Paux);

        if (i+1 < P.dimension(0)) {
          Tensor<elt_t> t5 = P(range(i,i+1,2), range(j));
          EXPECT_EQ(t5, t);
          unchanged(P, Paux);
        }
      }
    }
    Tensor<elt_t> t = P(range(), range());
    EXPECT_EQ(t, P);
    unchanged(P, Paux);
  }

  template<typename elt_t> void test_range(Tensor<elt_t> &P) {
    switch (P.rank()) {
    case 1: test_range1(P); break;
    case 2: test_range2(P); break;
    }
  }

  /////////////////////////////////////////////////////////////////////
  // REAL SPECIALIZATIONS
  //

  TEST(SliceTest, SliceRTensorTest) {
    test_over_tensors<double>(test_range<double>);
  }

  /////////////////////////////////////////////////////////////////////
  // COMPLEX SPECIALIZATIONS
  //

  TEST(SliceTest, SliceCTensorTest) {
    test_over_tensors<cdouble>(test_range<cdouble>);
  }

} // namespace tensor_test
