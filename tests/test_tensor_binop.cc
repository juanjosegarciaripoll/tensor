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

static constexpr int default_repeats = 10;

// Test binary operations among tensors
//
template <typename elt_t, typename elt_t2, typename elt_t3>
void test_tensor_tensor_binop(Tensor<elt_t> &P) {
  const Tensor<elt_t> Pcopy(P);
  elt_t orig;
  if (P.size()) orig = Pcopy[0];
  for (size_t times = 0; times < default_repeats; times++) {
    auto Paux = Tensor<elt_t2>::random(P.dimensions());
    const Tensor<elt_t3> P1 = P + Paux;
    const Tensor<elt_t3> P2 = P - Paux;
    const Tensor<elt_t3> P3 = P * Paux;
    const Tensor<elt_t3> P4 = P / Paux;
    EXPECT_TRUE(all_equal(P1.dimensions(), P.dimensions()));
    EXPECT_TRUE(all_equal(P2.dimensions(), P.dimensions()));
    EXPECT_TRUE(all_equal(P3.dimensions(), P.dimensions()));
    EXPECT_TRUE(all_equal(P4.dimensions(), P.dimensions()));
    for (size_t i = 0; i < P.size(); i++) {
      ASSERT_EQ(P1[i], P[i] + Paux[i])
          << "P[i]=" << P[i] << ", Paux[i]=" << Paux[i];
      ASSERT_EQ(P2[i], P[i] - Paux[i])
          << "P[i]=" << P[i] << ", Paux[i]=" << Paux[i];
      ASSERT_EQ(P3[i], P[i] * Paux[i])
          << "P[i]=" << P[i] << ", Paux[i]=" << Paux[i];
      ASSERT_EQ(P4[i], P[i] / Paux[i])
          << "P[i]=" << P[i] << ", Paux[i]=" << Paux[i];
    }
    unchanged(P, Pcopy);
    if (P.size()) EXPECT_EQ(Pcopy[0], orig);
  }
}

// Test binary operations among tensors and numbers
//
template <typename elt_t, typename elt_t2, typename elt_t3>
void test_tensor_number_binop(Tensor<elt_t> &P) {
  const Tensor<elt_t> Pcopy(P);
  elt_t orig;
  if (P.size()) orig = Pcopy[0];
  for (size_t times = 0; times < default_repeats; times++) {
    auto aux = rand<elt_t2>();
    const Tensor<elt_t3> P1 = P + aux;
    const Tensor<elt_t3> P2 = P - aux;
    const Tensor<elt_t3> P3 = P * aux;
    const Tensor<elt_t3> P4 = P / aux;
    EXPECT_TRUE(all_equal(P1.dimensions(), P.dimensions()));
    EXPECT_TRUE(all_equal(P2.dimensions(), P.dimensions()));
    EXPECT_TRUE(all_equal(P3.dimensions(), P.dimensions()));
    EXPECT_TRUE(all_equal(P4.dimensions(), P.dimensions()));
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
template <typename elt_t, typename elt_t2, typename elt_t3>
void test_number_tensor_binop(Tensor<elt_t> &P) {
  const Tensor<elt_t> Pcopy(P);
  elt_t orig;
  if (P.size()) orig = Pcopy[0];
  for (size_t times = 0; times < default_repeats; times++) {
    auto aux = rand<elt_t2>();
    const Tensor<elt_t3> P1 = aux + P;
    const Tensor<elt_t3> P2 = aux - P;
    const Tensor<elt_t3> P3 = aux * P;
    const Tensor<elt_t3> P4 = aux / P;
    EXPECT_TRUE(all_equal(P1.dimensions(), P.dimensions()));
    EXPECT_TRUE(all_equal(P2.dimensions(), P.dimensions()));
    EXPECT_TRUE(all_equal(P3.dimensions(), P.dimensions()));
    EXPECT_TRUE(all_equal(P4.dimensions(), P.dimensions()));
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

//
// IN-PLACE UPDATES
//

template <typename elt_t, typename elt_t2, typename elt_t3>
void test_tensor_tensor_binop_in_place(Tensor<elt_t> &P) {
  for (size_t times = 0; times < default_repeats; times++) {
    {
      auto P1 = P;
      auto P2 = Tensor<elt_t2>::random(P.dimensions());
      P1 += P2;
      EXPECT_TRUE(all_equal(P1.dimensions(), P.dimensions()));
      for (size_t i = 0; i < P1.size(); i++) {
        ASSERT_EQ(P1[i], P[i] + P2[i]);
      }
    }
    {
      auto P1 = P;
      auto P2 = Tensor<elt_t2>::random(P.dimensions());
      P1 -= P2;
      EXPECT_TRUE(all_equal(P1.dimensions(), P.dimensions()));
      for (size_t i = 0; i < P1.size(); i++) {
        ASSERT_EQ(P1[i], P[i] - P2[i]);
      }
    }
    {
      auto P1 = P;
      auto P2 = Tensor<elt_t2>::random(P.dimensions());
      P1 *= P2;
      EXPECT_TRUE(all_equal(P1.dimensions(), P.dimensions()));
      for (size_t i = 0; i < P1.size(); i++) {
        ASSERT_EQ(P1[i], P[i] * P2[i]);
      }
    }
    {
      auto P1 = P;
      const double large_offset = 2.0;
      auto P2 = large_offset + Tensor<elt_t2>::random(P.dimensions());
      P1 /= P2;
      EXPECT_TRUE(all_equal(P1.dimensions(), P.dimensions()));
      for (size_t i = 0; i < P1.size(); i++) {
        ASSERT_EQ(P1[i], P[i] / P2[i]);
      }
    }
  }
}

template <typename elt_t, typename elt_t2, typename elt_t3>
void test_tensor_number_binop_in_place(Tensor<elt_t> &P) {
  for (size_t times = 0; times < default_repeats; times++) {
    {
      auto P1 = P;
      auto P2 = rand<elt_t2>();
      P1 += P2;
      EXPECT_TRUE(all_equal(P1.dimensions(), P.dimensions()));
      for (size_t i = 0; i < P1.size(); i++) {
        ASSERT_EQ(P1[i], P[i] + P2);
      }
    }
    {
      auto P1 = P;
      auto P2 = rand<elt_t2>();
      P1 -= P2;
      EXPECT_TRUE(all_equal(P1.dimensions(), P.dimensions()));
      for (size_t i = 0; i < P1.size(); i++) {
        ASSERT_EQ(P1[i], P[i] - P2);
      }
    }
    {
      auto P1 = P;
      auto P2 = rand<elt_t2>();
      P1 *= P2;
      EXPECT_TRUE(all_equal(P1.dimensions(), P.dimensions()));
      for (size_t i = 0; i < P1.size(); i++) {
        ASSERT_EQ(P1[i], P[i] * P2);
      }
    }
    {
      auto P1 = P;
      const double large_offset = 2.0;
      auto P2 = large_offset + rand<elt_t2>();
      P1 /= P2;
      EXPECT_TRUE(all_equal(P1.dimensions(), P.dimensions()));
      for (size_t i = 0; i < P1.size(); i++) {
        ASSERT_EQ(P1[i], P[i] / P2);
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

TEST(TensorBinopTest, RTensorRTensorBinop) {
  test_over_tensors<double>(test_tensor_tensor_binop<double, double, double>);
}

TEST(TensorBinopTest, RTensorDoubleBinop) {
  test_over_tensors<double>(test_tensor_number_binop<double, double, double>);
}

TEST(TensorBinopTest, DoubleRTensorBinop) {
  test_over_tensors<double>(test_number_tensor_binop<double, double, double>);
}

TEST(TensorBinopTest, RTensorRTensorBinopInPlace) {
  test_over_tensors<double>(
      test_tensor_tensor_binop_in_place<double, double, double>);
}

TEST(TensorBinopTest, RTensorDoubleBinopInPlace) {
  test_over_tensors<double>(
      test_tensor_number_binop_in_place<double, double, double>);
}

//////////////////////////////////////////////////////////////////////
// COMPLEX SPECIALIZATIONS
//

TEST(TensorBinopTest, CTensorCTensorBinop) {
  test_over_tensors<cdouble>(
      test_tensor_tensor_binop<cdouble, cdouble, cdouble>);
}

TEST(TensorBinopTest, CTensorCdoubleBinop) {
  test_over_tensors<cdouble>(
      test_tensor_number_binop<cdouble, cdouble, cdouble>);
}

TEST(TensorBinopTest, CdoubleCTensorBinop) {
  test_over_tensors<cdouble>(
      test_number_tensor_binop<cdouble, cdouble, cdouble>);
}

TEST(TensorBinopTest, CTensorDoubleBinop) {
  test_over_tensors<cdouble>(
      test_tensor_number_binop<cdouble, double, cdouble>);
}

TEST(TensorBinopTest, CTensorRTensorBinop) {
  test_over_tensors<cdouble>(
      test_tensor_tensor_binop<cdouble, double, cdouble>);
}

TEST(TensorBinopTest, CdoubleRTensorBinop) {
  test_over_tensors<cdouble>(
      test_number_tensor_binop<cdouble, double, cdouble>);
}

TEST(TensorBinopTest, DoubleCTensorBinop) {
  test_over_tensors<double>(test_number_tensor_binop<double, cdouble, cdouble>);
}

TEST(TensorBinopTest, RTensorCTensorBinop) {
  test_over_tensors<double>(test_tensor_tensor_binop<double, cdouble, cdouble>);
}

TEST(TensorBinopTest, RTensorCdoubleBinop) {
  test_over_tensors<double>(test_tensor_number_binop<double, cdouble, cdouble>);
}

TEST(TensorBinopTest, CTensorCTensorBinopInPlace) {
  test_over_tensors<cdouble>(
      test_tensor_tensor_binop_in_place<cdouble, cdouble, cdouble>);
}

TEST(TensorBinopTest, CTensorCdoubleBinopInPlace) {
  test_over_tensors<cdouble>(
      test_tensor_number_binop_in_place<cdouble, cdouble, cdouble>);
}

TEST(TensorBinopTest, CTensorRTensorBinopInPlace) {
  test_over_tensors<cdouble>(
      test_tensor_tensor_binop_in_place<cdouble, double, cdouble>);
}

TEST(TensorBinopTest, CTensorDoubleBinopInPlace) {
  test_over_tensors<cdouble>(
      test_tensor_number_binop_in_place<cdouble, double, cdouble>);
}

TEST(TensorBinopTest, PowRTensorRTensor) {
  EXPECT_CEQ(::tensor::pow(RTensor{1.0, -1.0, 2.0, -2.0},
                           RTensor{2.0, -2.0, 3.0, -3.0}),
             RTensor({1, 1, 8.0, -1 / 8.0}));
}

TEST(TensorBinopTest, PowCTensorRTensor) {
  EXPECT_CEQ(::tensor::pow(CTensor{cdouble(0.0, 1.0), cdouble(0.0, 1.0),
                                   cdouble(0.0, 2.0), cdouble(0.0, -2.0)},
                           RTensor{-1.0, -2.0, 3.0, -3.0}),
             CTensor({cdouble(0.0, -1.0), cdouble(-1.0, 0.0),
                      cdouble(0.0, -8.0), cdouble(0.0, -1 / 8.0)}));
}

}  // namespace tensor_test
