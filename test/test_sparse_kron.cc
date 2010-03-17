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
  // SIZES OF KRONECKER PRODUCTS
  //

  template<typename elt_t>
  void do_test_kron_size(Tensor<elt_t> &a, Tensor<elt_t> &b) {
    Sparse<elt_t> sa = Sparse<elt_t>::random(a.rows(), a.columns());
    Sparse<elt_t> sb = Sparse<elt_t>::random(b.rows(), b.columns());

    Sparse<elt_t> sk = kron(sa, sb);

    ASSERT_EQ(sk.rows(), sa.rows() * sb.rows());
    ASSERT_EQ(sk.columns(), sa.columns() * sb.columns());
    ASSERT_EQ(sk.length(), sa.length() * sb.length());

    Sparse<elt_t> sk2 = kron2(sa, sb);
    ASSERT_EQ(sk2.rows(), sa.rows() * sb.rows());
    ASSERT_EQ(sk2.columns(), sa.columns() * sb.columns());
    ASSERT_EQ(sk2.length(), sa.length() * sb.length());
  }

  TEST(RSparseKronTest, KronSize) {
    test_over_fixed_rank_pairs<double>(do_test_kron_size<double>, 2);
  }

  TEST(CSparseKronTest, KronSize) {
    test_over_fixed_rank_pairs<cdouble>(do_test_kron_size<cdouble>, 2);
  }

} // namespace tensor_test
