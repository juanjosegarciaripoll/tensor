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

template <typename elt_t>
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

template <typename elt_t>
void test_kron_small() {
  kron_2d_fixture<elt_t> fixture;

  for (auto it = fixture.begin(); it != fixture.end();) {
    Tensor<elt_t> a(*(it++));
    Tensor<elt_t> b(*(it++));
    Tensor<elt_t> k(*(it++));

    ASSERT_TRUE(all_equal(k, kron(a, b)));
    ASSERT_TRUE(all_equal(kron(b, a), kron2(a, b)));
  }
}

TEST(RTensorKronTest, KronSmall) { test_kron_small<double>(); }

TEST(CTensorKronTest, KronSmall) { test_kron_small<cdouble>(); }

//
// COMPARISON WITH SLOW FORMULAS
//

template <typename elt_t>
Tensor<elt_t> slow_kron(const Tensor<elt_t> &a, const Tensor<elt_t> &b) {
  tensor::index a1, a2, b1, b2;
  a.get_dimensions(&a1, &a2);
  b.get_dimensions(&b1, &b2);

  if (a1 == 0 || a2 == 0 || b1 == 0 || b2 == 0)
    return Tensor<elt_t>::empty(a1 * b1, b2 * a2);

  auto output = Tensor<elt_t>::empty(b1, a1, b2, a2);
  for (tensor::index i = 0; i < b1; i++)
    for (tensor::index j = 0; j < a1; j++)
      for (tensor::index k = 0; k < b2; k++)
        for (tensor::index l = 0; l < a2; l++)
          output.at(i, j, k, l) = b(i, k) * a(j, l);
  return reshape(output, b1 * a1, b2 * a2);
}

template <typename elt_t>
void test_slow_kron(Tensor<elt_t> &a, Tensor<elt_t> &b) {
  a.randomize();
  b.randomize();
  ASSERT_TRUE(all_equal(slow_kron(a, b), kron(a, b)));
}

TEST(RTensorKronTest, CompareWithSlowKron) {
  test_over_fixed_rank_pairs<double>(test_slow_kron<double>, 2);
}

TEST(CTensorKronTest, CompareWithSlowKron) {
  test_over_fixed_rank_pairs<cdouble>(test_slow_kron<cdouble>, 2);
}

}  // namespace tensor_test
