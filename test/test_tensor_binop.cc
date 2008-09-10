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
  void test_tensor_tensor_binop(Tensor<elt_t> &P)
  {
    const Tensor<elt_t> Pcopy(P);
    elt_t orig;
    if (P.size()) orig = Pcopy[0];
    for (size_t times = 0; times < 10; times++) {
	Tensor<elt_t2> Paux(P.dimensions());
	Paux.randomize();
	const Tensor<elt_t3> P1 = P + Paux;
	const Tensor<elt_t3> P2 = P - Paux;
	const Tensor<elt_t3> P3 = P * Paux;
	const Tensor<elt_t3> P4 = P / Paux;
	EXPECT_EQ(P1.dimensions(), P.dimensions());
	EXPECT_EQ(P2.dimensions(), P.dimensions());
	EXPECT_EQ(P3.dimensions(), P.dimensions());
	EXPECT_EQ(P4.dimensions(), P.dimensions());
	for (size_t i = 0; i < P.size(); i++) {
	  ASSERT_EQ(P1[i], P[i] + Paux[i]) << "P[i]=" << P[i] << ", Paux[i]=" << Paux[i];
	  ASSERT_EQ(P2[i], P[i] - Paux[i]) << "P[i]=" << P[i] << ", Paux[i]=" << Paux[i];
	  ASSERT_EQ(P3[i], P[i] * Paux[i]) << "P[i]=" << P[i] << ", Paux[i]=" << Paux[i];
	  ASSERT_EQ(P4[i], P[i] / Paux[i]) << "P[i]=" << P[i] << ", Paux[i]=" << Paux[i];
	}
	unchanged(P, Pcopy);
	if (P.size()) EXPECT_EQ(Pcopy[0], orig);
    }
  }

  // Test binary operations among tensors and numbers
  //
  template<typename elt_t, typename elt_t2, typename elt_t3>
  void test_tensor_number_binop(Tensor<elt_t> &P)
  {
    const Tensor<elt_t> Pcopy(P);
    elt_t orig;
    if (P.size()) orig = Pcopy[0];
    for (size_t times = 0; times < 10; times++) {
	elt_t2 aux = rand<elt_t2>();
	const Tensor<elt_t3> P1 = P + aux;
	const Tensor<elt_t3> P2 = P - aux;
	const Tensor<elt_t3> P3 = P * aux;
	const Tensor<elt_t3> P4 = P / aux;
	EXPECT_EQ(P1.dimensions(), P.dimensions());
	EXPECT_EQ(P2.dimensions(), P.dimensions());
	EXPECT_EQ(P3.dimensions(), P.dimensions());
	EXPECT_EQ(P4.dimensions(), P.dimensions());
	for (size_t i = 0; i < P.size(); i++) {
	  ASSERT_EQ(P1[i], P[i] + aux) << "P[i]=" << P[i] << ", aux=" << aux;
	  ASSERT_EQ(P2[i], P[i] - aux) << "P[i]=" << P[i] << ", aux=" << aux;
	  ASSERT_EQ(P3[i], P[i] * aux) << "P[i]=" << P[i] << ", aux=" << aux;
	  ASSERT_EQ(P4[i], P[i] / aux) << "P[i]=" << P[i] << ", aux=" << aux;
	}
	unchanged(P, Pcopy);
	if (P.size()) EXPECT_EQ(Pcopy[0], orig);
    }
  }

  // Test binary operations among tensors and numbers
  //
  template<typename elt_t, typename elt_t2, typename elt_t3>
  void test_number_tensor_binop(Tensor<elt_t> &P)
  {
    const Tensor<elt_t> Pcopy(P);
    elt_t orig;
    if (P.size()) orig = Pcopy[0];
    for (size_t times = 0; times < 10; times++) {
	elt_t2 aux = rand<elt_t2>();
	const Tensor<elt_t3> P1 = aux + P;
	const Tensor<elt_t3> P2 = aux - P;
	const Tensor<elt_t3> P3 = aux * P;
	const Tensor<elt_t3> P4 = aux / P;
	EXPECT_EQ(P1.dimensions(), P.dimensions());
	EXPECT_EQ(P2.dimensions(), P.dimensions());
	EXPECT_EQ(P3.dimensions(), P.dimensions());
	EXPECT_EQ(P4.dimensions(), P.dimensions());
	for (size_t i = 0; i < P.size(); i++) {
	  ASSERT_EQ(P1[i], aux + P[i]) << "P[i]=" << P[i] << ", aux=" << aux;
	  ASSERT_EQ(P2[i], aux - P[i]) << "P[i]=" << P[i] << ", aux=" << aux;
	  ASSERT_EQ(P3[i], aux * P[i]) << "P[i]=" << P[i] << ", aux=" << aux;
	  ASSERT_EQ(P4[i], aux / P[i]) << "P[i]=" << P[i] << ", aux=" << aux;
	}
	unchanged(P, Pcopy);
	if (P.size()) EXPECT_EQ(Pcopy[0], orig);
    }
  }

  //////////////////////////////////////////////////////////////////////
  // REAL SPECIALIZATIONS
  //

  TEST(TensorBinopTest, RTensorRTensorBinop) {
    test_over_tensors<double>(test_tensor_tensor_binop<double,double,double>);
  }

  TEST(TensorBinopTest, RTensorDoubleBinop) {
    test_over_tensors<double>(test_tensor_number_binop<double,double,double>);
  }

  TEST(TensorBinopTest, DoubleRTensorBinop) {
    test_over_tensors<double>(test_number_tensor_binop<double,double,double>);
  }

  //////////////////////////////////////////////////////////////////////
  // COMPLEX SPECIALIZATIONS
  //

  TEST(TensorBinopTest, CTensorCTensorBinop) {
    test_over_tensors<cdouble>(test_tensor_tensor_binop<cdouble,cdouble,cdouble>);
  }

  TEST(TensorBinopTest, CTensorCdoubleBinop) {
    test_over_tensors<cdouble>(test_tensor_number_binop<cdouble,cdouble,cdouble>);
  }

  TEST(TensorBinopTest, CdoubleCTensorBinop) {
    test_over_tensors<cdouble>(test_number_tensor_binop<cdouble,cdouble,cdouble>);
  }

  TEST(TensorBinopTest, CTensorDoubleBinop) {
    test_over_tensors<cdouble>(test_tensor_number_binop<cdouble,double,cdouble>);
  }

  TEST(TensorBinopTest, CTensorRTensorBinop) {
    test_over_tensors<cdouble>(test_tensor_tensor_binop<cdouble,double,cdouble>);
  }

  TEST(TensorBinopTest, CdoubleRTensorBinop) {
    test_over_tensors<cdouble>(test_number_tensor_binop<cdouble,double,cdouble>);
  }

  TEST(TensorBinopTest, DoubleCTensorBinop) {
    test_over_tensors<double>(test_number_tensor_binop<double,cdouble,cdouble>);
  }

  TEST(TensorBinopTest, RTensorCTensorBinop) {
    test_over_tensors<double>(test_tensor_tensor_binop<double,cdouble,cdouble>);
  }

  TEST(TensorBinopTest, RTensorCdoubleBinop) {
    test_over_tensors<double>(test_tensor_number_binop<double,cdouble,cdouble>);
  }

} // namespace tensor_test
