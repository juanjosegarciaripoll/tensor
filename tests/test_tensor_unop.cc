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

////////////////////////////////////////////////////////////////////////
//
// WRAPPERS FOR SOLVING SOME PROBLEMS WITH AIX
//
template <typename elt_t>
double _abs(elt_t x) {
  return tensor::abs(x);
}
template <typename elt_t>
elt_t _exp(elt_t x) {
  return exp(x);
}
template <typename elt_t>
elt_t _sin(elt_t x) {
  return sin(x);
}
template <typename elt_t>
elt_t _cos(elt_t x) {
  return cos(x);
}
template <typename elt_t>
elt_t _tan(elt_t x) {
  return tan(x);
}
template <typename elt_t>
elt_t _sinh(elt_t x) {
  return sinh(x);
}
template <typename elt_t>
elt_t _cosh(elt_t x) {
  return cosh(x);
}
template <typename elt_t>
elt_t _tanh(elt_t x) {
  return tanh(x);
}

////////////////////////////////////////////////////////////////////////
//
// TESTING TENSOR UNARY OPERATIONS
//

template <typename elt_t, typename elt_t2, elt_t2 f(elt_t),
          Tensor<elt_t2> fT(const Tensor<elt_t> &)>
void test_unop(Tensor<elt_t> &P) {
  const Tensor<elt_t> Pcopy(P);
  Tensor<elt_t2> P2 = fT(P);
  unchanged(P, Pcopy);
  unique(P2);
  EXPECT_TRUE(all_equal(P.dimensions(), P2.dimensions()));
  for (tensor::index i = 0; i < P.size(); i++) {
    ASSERT_EQ(f(P[i]), P2[i]);
  }
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

TEST(TensorUnaryOperatorTest, RTensorAbs) {
  test_over_tensors<double>(test_unop<double, double, _abs, abs>, 6, 4, 30);
}
TEST(TensorUnaryOperatorTest, RTensorExp) {
  test_over_tensors<double>(test_unop<double, double, _exp, exp>, 6, 4, 30);
}
TEST(TensorUnaryOperatorTest, RTensorSin) {
  test_over_tensors<double>(test_unop<double, double, _sin, sin>, 6, 4, 30);
}
TEST(TensorUnaryOperatorTest, RTensorCos) {
  test_over_tensors<double>(test_unop<double, double, _cos, cos>, 6, 4, 30);
}
TEST(TensorUnaryOperatorTest, RTensorTan) {
  test_over_tensors<double>(test_unop<double, double, _tan, tan>, 6, 4, 30);
}
TEST(TensorUnaryOperatorTest, RTensorSinh) {
  test_over_tensors<double>(test_unop<double, double, _sinh, sinh>, 6, 4, 30);
}
TEST(TensorUnaryOperatorTest, RTensorCosh) {
  test_over_tensors<double>(test_unop<double, double, _cosh, cosh>, 6, 4, 30);
}
TEST(TensorUnaryOperatorTest, RTensorTanh) {
  test_over_tensors<double>(test_unop<double, double, _tanh, tanh>, 6, 4, 30);
}

//////////////////////////////////////////////////////////////////////
// COMPLEX SPECIALIZATIONS
//

TEST(TensorUnaryOperatorTest, CTensorAbs) {
  test_over_tensors<cdouble>(test_unop<cdouble, double, _abs, abs>, 6, 4, 30);
}
TEST(TensorUnaryOperatorTest, CTensorExp) {
  test_over_tensors<cdouble>(test_unop<cdouble, cdouble, _exp, exp>, 6, 4, 30);
}
TEST(TensorUnaryOperatorTest, CTensorSin) {
  test_over_tensors<cdouble>(test_unop<cdouble, cdouble, _sin, sin>, 6, 4, 30);
}
TEST(TensorUnaryOperatorTest, CTensorCos) {
  test_over_tensors<cdouble>(test_unop<cdouble, cdouble, _cos, cos>, 6, 4, 30);
}
TEST(TensorUnaryOperatorTest, CTensorTan) {
  test_over_tensors<cdouble>(test_unop<cdouble, cdouble, _tan, tan>, 6, 4, 30);
}
TEST(TensorUnaryOperatorTest, CTensorSinh) {
  test_over_tensors<cdouble>(test_unop<cdouble, cdouble, _sinh, sinh>, 6, 4,
                             30);
}
TEST(TensorUnaryOperatorTest, CTensorCosh) {
  test_over_tensors<cdouble>(test_unop<cdouble, cdouble, _cosh, cosh>, 6, 4,
                             30);
}
TEST(TensorUnaryOperatorTest, CTensorTanh) {
  test_over_tensors<cdouble>(test_unop<cdouble, cdouble, _tanh, tanh>, 6, 4,
                             30);
}

}  // namespace tensor_test
