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

template <typename elt_t>
void test_take_diag(int n) {
  for (int i = -n + 1; i < n; i++) {
    int l = n - abs(i);
    Tensor<elt_t> orig = Tensor<elt_t>::random(l);
    Tensor<elt_t> m = diag(orig, i);
    Tensor<elt_t> other = take_diag(m, i);
    EXPECT_TRUE(all_equal(orig, other));
  }
}

TEST(RTensorTest, SimpleTakeDiagTest) {
  {
    RTensor m(igen << 2 << 2);
    m.at(0, 0) = 1.0;
    m.at(0, 1) = 2.0;
    m.at(1, 0) = 3.0;
    m.at(1, 1) = 4.0;
    RTensor d0(igen << 2, rgen << 1.0 << 4.0);
    RTensor d1(igen << 1, rgen << 2.0);
    RTensor dm1(igen << 1, rgen << 3.0);
    EXPECT_TRUE(all_equal(d0, take_diag(m, 0)));
    EXPECT_TRUE(all_equal(d1, take_diag(m, 1)));
    EXPECT_TRUE(all_equal(dm1, take_diag(m, -1)));
  }
  {
    RTensor m(igen << 2 << 3);
    m.at(0, 0) = 1.0;
    m.at(0, 1) = 2.0;
    m.at(0, 2) = 3.0;
    m.at(1, 0) = 4.0;
    m.at(1, 1) = 5.0;
    m.at(1, 2) = 6.0;
    RTensor d0(igen << 2, rgen << 1.0 << 5.0);
    RTensor d1(igen << 2, rgen << 2.0 << 6.0);
    RTensor d2(igen << 1, rgen << 3.0);
    RTensor dm1(igen << 1, rgen << 4.0);
    EXPECT_TRUE(all_equal(d0, take_diag(m, 0)));
    EXPECT_TRUE(all_equal(d1, take_diag(m, 1)));
    EXPECT_TRUE(all_equal(d2, take_diag(m, 2)));
    EXPECT_TRUE(all_equal(dm1, take_diag(m, -1)));
  }
}

TEST(RTensorTest, TakeDiagTest) {
  test_over_integers(0, 10, test_take_diag<double>);
}

TEST(CTensorTest, TakeDiagTest) {
  test_over_integers(0, 10, test_take_diag<cdouble>);
}

}  // namespace tensor_test
