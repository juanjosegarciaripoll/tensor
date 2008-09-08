// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include "loops.h"
#include <gtest/gtest.h>
#include <tensor/tensor.h>

namespace tensor_test {

  template<typename elt_t> void test_empty_constructor() {
    Tensor<elt_t> P;
    EXPECT_EQ(0, P.size());
    EXPECT_EQ(0, P.rank());
    EXPECT_EQ(0, P.begin_const());
  }

  template<typename elt_t> void test_copy_constructor(Tensor<elt_t> &P) {
    Tensor<elt_t> P2(P);
    unchanged(P2, P);
  }

  // Test proper work of dimension querying routines.
  //
  template<typename elt_t> void test_dims(Tensor<elt_t> &P) {
    Indices d = P.dimensions();
    Indices::elt_t a1,a2,a3,a4,a5,a6;
    ASSERT_LE(P.rank(), 6);
    EXPECT_EQ(d.size(), P.rank());
    switch (P.rank()) {
    case 6: {
      P.get_dimensions(&a1,&a2,&a3,&a4,&a5,&a6);
      EXPECT_EQ(a1*a2*a3*a4*a5*a6, P.size());
      break;
    }
    case 5: {
      P.get_dimensions(&a1,&a2,&a3,&a4,&a5);
      EXPECT_EQ(a1*a2*a3*a4*a5, P.size());
      break;
    }
    case 4: {
      P.get_dimensions(&a1,&a2,&a3,&a4);
      EXPECT_EQ(a1*a2*a3*a4, P.size());
      break;
    }
    case 3: {
      P.get_dimensions(&a1,&a2,&a3);
      EXPECT_EQ(a1*a2*a3, P.size());
      break;
    }
    case 2: {
      P.get_dimensions(&a1,&a2);
      EXPECT_EQ(a1*a2, P.size());
      break;
    }
    case 1: {
      P.get_dimensions(&a1);
      EXPECT_EQ(a1, P.size());
      break;
    }
    case 0: {
      EXPECT_EQ(0, P.size());
      break;
    }
    }
  }

  template<typename elt_t> void test_constructor_with_dims(Tensor<elt_t> &P) {
    Indices d = P.dimensions();
    {
      Tensor<elt_t> P2(d);
      EXPECT_EQ(P.rank(), P2.rank());
      EXPECT_EQ(P.size(), P2.size());
      EXPECT_TRUE(d == P2.dimensions());
    }
    {
      Tensor<elt_t> P2;
      // Test constructors for tensors of given rank
      switch (P.rank()) {
      case 6: { P2 = Tensor<elt_t>(d[0],d[1],d[2],d[3],d[4],d[5]); break; }
      case 5: { P2 = Tensor<elt_t>(d[0],d[1],d[2],d[3],d[4]); break; }
      case 4: { P2 = Tensor<elt_t>(d[0],d[1],d[2],d[3]); break; }
      case 3: { P2 = Tensor<elt_t>(d[0],d[1],d[2]); break; }
      case 2: { P2 = Tensor<elt_t>(d[0],d[1]); break; }
      case 1: { P2 = Tensor<elt_t>(d[0]); break; }
      case 0: { P2 = Tensor<elt_t>(); break; }
      }
      EXPECT_EQ(P.rank(), P2.rank());
      EXPECT_EQ(P.size(), P2.size());
      EXPECT_TRUE(d == P2.dimensions());
    }
  }

  // Verify that we can create a copy of a tensor with different
  // dimensions, but having the same data.
  template<typename elt_t>
  void test_copy_constructor_with_dims(Tensor<elt_t> &P) {
    Indices d = P.dimensions();
    for (int i = 0; i < P.rank(); i++) {
      Indices copy(i+1);
      for (int j = 0; j < P.rank(); j++) {
	if (j > i) {
	  copy.at(i) *= d[j];
	} else {
	  copy.at(j) = d[j];
	}
      }
      Tensor<elt_t> P2(copy, P);
      EXPECT_EQ(i+1, P2.rank());
      EXPECT_EQ(P.size(), P2.size());
      EXPECT_EQ(copy, P2.dimensions());
      unchanged(P2, P);
    }
  }

  //////////////////////////////////////////////////////////////////////
  // REAL SPECIALIZATIONS
  //

  TEST(RTensorTest, RTensorEmptyConstructor) {
    test_empty_constructor<double>();
  }

  TEST(RTensorTest, RTensorCopyConstructor) {
    test_over_tensors(test_copy_constructor<double>);
  }

  TEST(RTensorTest, RTensorDims) {
    test_over_tensors(test_dims<double>);
  }

  TEST(RTensorTest, RTensorCopyConstructorWithDims) {
    test_over_tensors(test_copy_constructor_with_dims<double>);
  }

  //////////////////////////////////////////////////////////////////////
  // COMPLEX SPECIALIZATIONS
  //

  TEST(CTensorTest, CTensorEmptyConstructor) {
    test_empty_constructor<cdouble>();
  }

  TEST(CTensorTest, CTensorCopyConstructor) {
    test_over_tensors(test_copy_constructor<cdouble>);
  }

  TEST(CTensorTest, CTensorDims) {
    test_over_tensors(test_dims<cdouble>);
  }

  TEST(CTensorTest, CTensorCopyConstructorWithDims) {
    test_over_tensors(test_copy_constructor_with_dims<cdouble>);
  }

} // namespace tensor_test
