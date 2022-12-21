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

////////////////////////////////////////////////////////////////////////
//
// TESTING ELEMENT RETRIEVAL AND SAFE COPYING
//

namespace tensor_test {

template <typename elt_t>
void store_nd(Tensor<elt_t> &P, size_t row_major_ndx, elt_t x) {
  const Indices &d = P.dimensions();
  size_t i1, i2, i3, i4;
  if (P.rank() == 1) {
    P.at(row_major_ndx) = x;
  } else if (P.rank() == 2) {
    i1 = row_major_ndx % d[0];
    i2 = row_major_ndx / d[0];
    P.at(i1, i2) = x;
  } else if (P.rank() == 3) {
    i1 = row_major_ndx % d[0];
    row_major_ndx /= d[0];
    i2 = row_major_ndx % d[1];
    i3 = row_major_ndx / d[1];
    P.at(i1, i2, i3) = x;
  } else if (P.rank() == 4) {
    i1 = row_major_ndx % d[0];
    row_major_ndx /= d[0];
    i2 = row_major_ndx % d[1];
    row_major_ndx /= d[1];
    i3 = row_major_ndx % d[2];
    i4 = row_major_ndx / d[2];
    P.at(i1, i2, i3, i4) = x;
  } else {
    std::cerr << "Tester does not support tensors with more than 4 dimensions";
    abort();
  }
}

template <typename elt_t>
elt_t get_nd(Tensor<elt_t> &P, size_t row_major_ndx) {
  const Indices &d = P.dimensions();
  size_t i1, i2, i3, i4;
  if (P.rank() == 1) {
    return P(row_major_ndx);
  } else if (P.rank() == 2) {
    i1 = row_major_ndx % d[0];
    i2 = row_major_ndx / d[0];
    return P(i1, i2);
  } else if (P.rank() == 3) {
    i1 = row_major_ndx % d[0];
    row_major_ndx /= d[0];
    i2 = row_major_ndx % d[1];
    i3 = row_major_ndx / d[1];
    return P(i1, i2, i3);
  } else if (P.rank() == 4) {
    i1 = row_major_ndx % d[0];
    row_major_ndx /= d[0];
    i2 = row_major_ndx % d[1];
    row_major_ndx /= d[1];
    i3 = row_major_ndx % d[2];
    i4 = row_major_ndx / d[2];
    return P(i1, i2, i3, i4);
  } else {
    std::cerr << "Tester does not support tensors with more than 4 dimensions";
    abort();
    return number_zero<elt_t>();
  }
}

// Equivalence between [] and (). Getter does not appropiate data.
template <typename elt_t>
void test_tensor_get(Tensor<elt_t> &P) {
  Tensor<elt_t> P2(P);
  for (tensor::index i = 0; i < P.size(); i++) {
    EXPECT_EQ(P[i], get_nd<elt_t>(P2, i));
    EXPECT_EQ(P[i], get_nd<elt_t>(P2, i));
  }
  unchanged(P2, P, 2);
}

// N-dimensional setter works and makes reference unique.
template <typename elt_t>
void test_tensor_set(Tensor<elt_t> &P) {
  Tensor<elt_t> P2(P);
  for (tensor::index i = 0; i < P.size(); i++) {
    elt_t x = P2[i] + number_one<elt_t>();
    ASSERT_NE(x, P2[i]);
    store_nd<elt_t>(P2, i, x);
    EXPECT_EQ(x, P2[i]);
    unique(P);
    unique(P2);
  }
}

// When we modify one of the copies, the other one remains intact.
template <typename elt_t>
void test_tensor_set_appropiates(Tensor<elt_t> &P) {
  auto modify_tensor = [](Tensor<elt_t> &P) {
    std::transform(P.begin(), P.end(), P.begin(),
                   [](elt_t x) { return x + number_one<elt_t>(); });
  };
  {
    typename Tensor<elt_t>::const_iterator old_p = P.cbegin();
    Tensor<elt_t> P2(P);
    unchanged(P2, P, 2);
    modify_tensor(P2);
    unique(P);
    unique(P2);
    EXPECT_EQ(old_p, P.cbegin());
    if (P.size()) {
      EXPECT_NE(old_p, P2.cbegin());
    }
  }
  {
    typename Tensor<elt_t>::const_iterator old_p = P.cbegin();
    Tensor<elt_t> P2(P);
    unchanged(P2, P, 2);
    modify_tensor(P);
    unique(P);
    unique(P2);
    EXPECT_EQ(old_p, P2.cbegin());
    if (P.size()) {
      EXPECT_NE(old_p, P.cbegin());
    }
  }
}

// When we modify one of the copies, only the first time results in
// memory allocation
template <typename elt_t>
void test_tensor_set_appropiates_only_once(Tensor<elt_t> &P) {
  if (P.size()) {
    Tensor<elt_t> P2(P);
    // Here P and P2 share memory
    unchanged(P, P2, 2);

    // Now P2 changed and unlinks from P
    P2.at_seq(0) = number_zero<elt_t>();
    unique(P);
    unique(P2);

    // But the second access does not cause new memory being allocated
    typename Tensor<elt_t>::const_iterator old_p = P2.cbegin();
    P2.at_seq(P2.size() - 1) = number_one<elt_t>();
    unique(P);
    unique(P2);
    EXPECT_EQ(old_p, P2.cbegin());
  }
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

TEST(RTensorTest, RTensorGet) { test_over_tensors(test_tensor_get<double>); }

TEST(RTensorTest, RTensorSet) { test_over_tensors(test_tensor_set<double>); }

TEST(RTensorTest, RTensorSetAppropiates) {
  test_over_tensors(test_tensor_set_appropiates<double>);
}

TEST(RTensorTest, RTensorSetAppropiatesOnlyOnce) {
  test_over_tensors(test_tensor_set_appropiates_only_once<double>);
}

//////////////////////////////////////////////////////////////////////
// COMPLEX SPECIALIZATIONS
//

TEST(CTensorTest, CTensorGet) { test_over_tensors(test_tensor_get<cdouble>); }

TEST(CTensorTest, CTensorSet) { test_over_tensors(test_tensor_set<cdouble>); }

TEST(CTensorTest, CTensorSetAppropiates) {
  test_over_tensors(test_tensor_set_appropiates<cdouble>);
}

TEST(CTensorTest, CTensorSetAppropiatesOnlyOnce) {
  test_over_tensors(test_tensor_set_appropiates_only_once<cdouble>);
}

}  // namespace tensor_test
