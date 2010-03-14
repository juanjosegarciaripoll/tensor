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

#include <tensor/sparse.h>
#include "loops.h"


#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>

namespace tensor_test {

  //
  // EMPTY SPARSE MATRICES
  //

  template<typename elt_t>
  void test_empty_constructor() {
    {
    SCOPED_TRACE("0x0");
    Sparse<elt_t> S;
    EXPECT_EQ(0, S.rows());
    EXPECT_EQ(0, S.columns());
    EXPECT_EQ(igen << 0, S.priv_row_start());
    //EXPECT_EQ(S.priv_row_start(), (igen << 0));
    EXPECT_EQ(0, S.priv_column().size());
    EXPECT_EQ(0, S.priv_data().size());
    EXPECT_EQ(full(S), Tensor<elt_t>(0,0));
    }
    {
    SCOPED_TRACE("0x1");
    Sparse<elt_t> S(0,2);
    EXPECT_EQ(0, S.rows());
    EXPECT_EQ(2, S.columns());
    EXPECT_EQ(igen << 0, S.priv_row_start());
    EXPECT_EQ(0, S.priv_column().size());
    EXPECT_EQ(0, S.priv_data().size());
    EXPECT_EQ(full(S), Tensor<elt_t>::zeros(0,2));
    }
    {
    SCOPED_TRACE("2x0");
    Sparse<elt_t> S(2,0);
    EXPECT_EQ(2, S.rows());
    EXPECT_EQ(0, S.columns());
    EXPECT_EQ(igen << 0 << 0 << 0, S.priv_row_start());
    EXPECT_EQ(0, S.priv_column().size());
    EXPECT_EQ(0, S.priv_data().size());
    EXPECT_EQ(full(S), Tensor<elt_t>::zeros(2,0));
    }
  }

  TEST(RSparseTest, RSparseEmptyConstructor) {
    test_empty_constructor<double>();
  }

  TEST(CSparseTest, CSparseEmptyConstructor) {
    test_empty_constructor<cdouble>();
  }

  //
  // SMALL SPARSE MATRICES
  //
  template<typename elt_t>
  void test_small_constructor() {
    {
    // sparse([1]);
    SCOPED_TRACE("1x1");
    Tensor<elt_t> T(igen << 1 << 1, 
                    gen<elt_t>(1.0));
    Sparse<elt_t> S(T);
    EXPECT_EQ(1, S.rows());
    EXPECT_EQ(1, S.columns());
    EXPECT_EQ(igen << 0 << 1, S.priv_row_start());
    EXPECT_EQ(igen << 0, S.priv_column());
    EXPECT_EQ(gen<elt_t>(1.0), S.priv_data());
    EXPECT_EQ(full(S), T);
    }
    {
    // sparse([1, 0; 0, 2]);
    SCOPED_TRACE("2x2");
    Tensor<elt_t> T(igen << 2 << 2, 
                    gen<elt_t>(1.0) << 0.0 << 0.0 << 2.0);
    Sparse<elt_t> S(T);
    EXPECT_EQ(2, S.rows());
    EXPECT_EQ(2, S.columns());
    EXPECT_EQ(igen << 0 << 1 << 2, S.priv_row_start());
    EXPECT_EQ(igen << 0 << 1, S.priv_column());
    EXPECT_EQ(gen<elt_t>(1.0) << 2.0, S.priv_data());
    EXPECT_EQ(full(S), T);
    }
    {
    // sparse([1, 2; 0, 3]);
    SCOPED_TRACE("2x2");
    Tensor<elt_t> T(igen << 2 << 2, 
                    gen<elt_t>(1.0) << 0.0 << 2.0 << 3.0);
    Sparse<elt_t> S(T);
    EXPECT_EQ(2, S.rows());
    EXPECT_EQ(2, S.columns());
    EXPECT_EQ(igen << 0 << 2 << 3, S.priv_row_start());
    EXPECT_EQ(igen << 0 << 1 << 1, S.priv_column());
    EXPECT_EQ(gen<elt_t>(1.0) << 2.0 << 3.0, S.priv_data());
    EXPECT_EQ(full(S), T);
    }
    {
    // sparse([1, 0, 4; 2, 3, 0]);
    SCOPED_TRACE("2x2");
    Tensor<elt_t> T(igen << 2 << 3, 
                    gen<elt_t>(1.0) << 2 << 0 << 3 << 4 << 0);
    Sparse<elt_t> S(T);
    EXPECT_EQ(2, S.rows());
    EXPECT_EQ(3, S.columns());
    EXPECT_EQ(igen << 0 << 2 << 4, S.priv_row_start());
    EXPECT_EQ(igen << 0 << 2 << 0 << 1, S.priv_column());
    EXPECT_EQ(gen<elt_t>(1.0) << 4.0 << 2.0 << 3.0, S.priv_data());
    EXPECT_EQ(full(S), T);
    }
  }

  TEST(RSparseTest, RSparseSmallConstructor) {
    test_small_constructor<double>();
  }

  TEST(CSparseTest, CSparseSmallConstructor) {
    test_small_constructor<cdouble>();
  }

} // namespace tensor_test
