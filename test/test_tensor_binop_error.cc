// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include "loops.h"
#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>
#include <tensor/tensor.h>

namespace tensor_test {

  // Test binary operations among tensors
  //
  template<typename elt_t, typename elt_t2, typename elt_t3>
  void test_tensor_tensor_binop_error(Tensor<elt_t> &P)
  {
#ifndef NDEBUG
    if (P.rank()) {
      {
	Tensor<elt_t2> Paux;
	EXPECT_EQ(0, Paux.rank());
	ASSERT_DEATH(P + Paux, ".*")
	  << P.dimensions() << "," << Paux.dimensions();
	ASSERT_DEATH(Paux + P, ".*")
	  << P.dimensions() << "," << Paux.dimensions();
      }
      {
	Indices dims = P.dimensions();
	dims.at(P.rank() - 1) += 1; 
	Tensor<elt_t2> Paux(dims);
	ASSERT_DEATH(P + Paux, ".*")
	  << P.dimensions() << "," << Paux.dimensions();
	ASSERT_DEATH(Paux + P, ".*")
	  << P.dimensions() << "," << Paux.dimensions();
      }
      {
	Tensor<elt_t2> Paux(P.dimension(0)+2);
	ASSERT_DEATH(Paux + P, ".*")
	  << P.dimensions() << "," << Paux.dimensions();
	ASSERT_DEATH(P + Paux, ".*")
	  << P.dimensions() << "," << Paux.dimensions();
      }
    }
#endif
  }


  //////////////////////////////////////////////////////////////////////
  // REAL SPECIALIZATIONS
  //

  TEST(TensorBinopTest, RTensorRTensorBinopError) {
    test_over_tensors<double>(test_tensor_tensor_binop_error<double,double,double>);
  }

  //////////////////////////////////////////////////////////////////////
  // COMPLEX SPECIALIZATIONS
  //

  TEST(TensorBinopTest, CTensorCTensorBinopError) {
    test_over_tensors<cdouble>(test_tensor_tensor_binop_error<cdouble,cdouble,cdouble>);
  }

  TEST(TensorBinopTest, CTensorRTensorBinopError) {
    test_over_tensors<cdouble>(test_tensor_tensor_binop_error<cdouble,double,cdouble>);
  }

  TEST(TensorBinopTest, RTensorCTensorBinopError) {
    test_over_tensors<double>(test_tensor_tensor_binop_error<double,cdouble,cdouble>);
  }

} // namespace tensor_test
