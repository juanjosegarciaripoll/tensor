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

// Test binary operations among tensors
//
template <typename elt_t, typename elt_t2, typename elt_t3>
void test_tensor_tensor_binop_error(Tensor<elt_t> &P) {
#ifndef NDEBUG
  if (P.size()) {
    {
      Tensor<elt_t2> Paux;
      EXPECT_EQ(0, Paux.rank());
      EXPECT_DEATH(P + Paux, "a.size.. == b.size..")
          << P.dimensions() << "," << Paux.dimensions();
      EXPECT_DEATH(Paux + P, ".*")
          << P.dimensions() << "," << Paux.dimensions();
    }
    {
      Indices dims = P.dimensions();
      dims.at(P.rank() - 1) += 1;
      Tensor<elt_t2> Paux(dims);
      EXPECT_DEATH(P + Paux, ".*")
          << P.dimensions() << "," << Paux.dimensions();
      EXPECT_DEATH(Paux + P, ".*")
          << P.dimensions() << "," << Paux.dimensions();
    }
    {
      Tensor<elt_t2> Paux = Tensor<elt_t2>::empty(P.dimension(0) + 1);
      EXPECT_DEATH(Paux + P, ".*")
          << P.dimensions() << "," << Paux.dimensions();
      EXPECT_DEATH(P + Paux, ".*")
          << P.dimensions() << "," << Paux.dimensions();
    }
  }
#endif
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

TEST(TensorBinopDeathTest, RTensorRTensorBinopError) {
  test_over_tensors<double>(
      test_tensor_tensor_binop_error<double, double, double>);
}

//////////////////////////////////////////////////////////////////////
// COMPLEX SPECIALIZATIONS
//

TEST(TensorBinopDeathTest, CTensorCTensorBinopError) {
  test_over_tensors<cdouble>(
      test_tensor_tensor_binop_error<cdouble, cdouble, cdouble>);
}

TEST(TensorBinopDeathTest, CTensorRTensorBinopError) {
  test_over_tensors<cdouble>(
      test_tensor_tensor_binop_error<cdouble, double, cdouble>);
}

TEST(TensorBinopDeathTest, RTensorCTensorBinopError) {
  test_over_tensors<double>(
      test_tensor_tensor_binop_error<double, cdouble, cdouble>);
}

}  // namespace tensor_test
