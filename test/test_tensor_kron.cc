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

#include <tensor/tensor.h>
#include "loops.h"
#include "test_kron.hpp"

namespace tensor_test {

  //
  // SIZES OF KRONECKER PRODUCTS
  //

  template<typename elt_t>
  void do_test_kron_size(Tensor<elt_t> &a, Tensor<elt_t> &b) {
    a.randomize();
    b.randomize();

    Tensor<elt_t> k = kron(a, b);

    ASSERT_EQ(k.rows(), a.rows() * b.rows());
    ASSERT_EQ(k.columns(), a.columns() * b.columns());
    ASSERT_EQ(k.size(), a.size() * b.size());

    Tensor<elt_t> k2 = kron2(a, b);
    ASSERT_EQ(k2.rows(), a.rows() * b.rows());
    ASSERT_EQ(k2.columns(), a.columns() * b.columns());
    ASSERT_EQ(k2.size(), a.size() * b.size());
  }

  TEST(RTensorKronTest, KronSize) {
    test_over_fixed_rank_pairs<double>(do_test_kron_size<double>, 2);
  }

  TEST(CTensorKronTest, KronSize) {
    test_over_fixed_rank_pairs<cdouble>(do_test_kron_size<cdouble>, 2);
  }

  //
  // HAND-BUILT KRONECKER PRODUCTS
  //

  template<typename elt_t>
  void test_kron_small() {
    kron_2d_fixture<elt_t> fixture;

    for(typename kron_2d_fixture<elt_t>::const_iterator it = fixture.begin();
        it != fixture.end();
        )
      {
        Tensor<elt_t> a(*(it++));
        Tensor<elt_t> b(*(it++));
        Tensor<elt_t> k(*(it++));

        ASSERT_EQ(k, kron(a, b));
        ASSERT_EQ(kron(b, a), kron2(a, b));
      }
  }

  TEST(RTensorKronTest, KronSmall) {
    test_kron_small<double>();
  }

  TEST(CTensorKronTest, KronSmall) {
    test_kron_small<cdouble>();
  }

} // namespace tensor_test
