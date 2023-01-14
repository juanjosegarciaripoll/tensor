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

using namespace tensor;
using tensor::index;

#include "test_view_common.cc"

//////////////////////////////////////////////////////////////////////
// MANUALLY SET RANGES FROM A TENSOR USING LOOPS
//

template <typename elt_t>
Tensor<elt_t> fill_continuous(const Tensor<elt_t> &P) {
  int m = 0;
  auto output = Tensor<elt_t>::empty(P.dimensions());
  for (typename Tensor<elt_t>::iterator it = output.begin(); it != output.end();
       ++it) {
    *it = number_zero<elt_t>() + (double)m++;
  }
  return output;
}

template <typename elt_t>
Tensor<elt_t> slow_range_set1(Tensor<elt_t> &P, index i0, index i2, index i1) {
  int n = 0;
  for (index i = i0, x = 0; i <= i2; i += i1, ++x, ++n) {
    P.at(x) = number_zero<elt_t>() + (double)n;
  }
}

template <typename elt_t>
Tensor<elt_t> slow_range_set2(const Tensor<elt_t> &P, index i0, index i2,
                              index i1, index j0, index j2, index j1) {
  int n = 0;
  for (index i = i0, x = 0; i <= i2; i += i1, ++x) {
    for (index j = j0, y = 0; j <= j2; j += j1, ++y) {
      P.at(x, y) = number_zero<elt_t>() + (double)n++;
    }
  }
}

template <typename elt_t>
void slow_range_set3(Tensor<elt_t> &P, index i0, index i2, index i1, index j0,
                     index j2, index j1, index k0, index k2, index k1) {
  int n = 0;
  for (index i = i0, x = 0; i <= i2; i += i1, ++x) {
    for (index j = j0, y = 0; j <= j2; j += j1, ++y) {
      for (index k = k0, z = 0; k <= k2; k += k1, ++z) {
        P.at(x, y, z) = number_zero<elt_t>() + (double)n++;
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////
// FULL RANGE ASSIGNMENT
//

template <typename elt_t>
void test_full_size_set1(Tensor<elt_t> &P) {
  Tensor<elt_t> t = fill_continuous(P);
  P.at(_) = t;
  ASSERT_TRUE(all_equal(P, t));
}

template <typename elt_t>
void test_full_size_set2(Tensor<elt_t> &P) {
  Tensor<elt_t> t = fill_continuous(P);
  P.at(_, _) = t;
  ASSERT_TRUE(all_equal(P, t));
}

template <typename elt_t>
void test_full_size_set3(Tensor<elt_t> &P) {
  Tensor<elt_t> t = fill_continuous(P);
  P.at(_, _, _) = t;
  ASSERT_TRUE(all_equal(P, t));
}

//
// REAL SPECIALIZATIONS
//

TEST(SliceSetTest, ImplicitCoercionToRange) {
  /*
   * This function compiles only when the first argument of a slicing method being
   * a range() forces the rest to be implicitely coerced to ranges.
   */
  RTensor aux{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  {
    RTensor aux_copy = aux.copy();
    aux_copy.at(range(0), Indices{0, 1, 2}) = RTensor{7, 8, 9};
    EXPECT_ALL_EQUAL(aux_copy.copy(),
                     RTensor({{7.0, 8.0, 9.0}, {4.0, 5.0, 6.0}}));
  }
  {
    RTensor aux_copy = aux.copy();
    aux_copy.at(_, range(0)) = RTensor({7.0, 8.0});
    EXPECT_ALL_EQUAL(aux_copy.copy(),
                     RTensor({{7.0, 2.0, 3.0}, {8.0, 5.0, 6.0}}));
  }
}

TEST(SliceSetTest, SliceRTensor1DFullSet) {
  test_over_fixed_rank_tensors<double>(test_full_size_set1<double>, 1);
}

TEST(SliceSetTest, SliceRTensor2DFullSet) {
  test_over_fixed_rank_tensors<double>(test_full_size_set2<double>, 2);
}

TEST(SliceSetTest, SliceRTensor3DFullSet) {
  test_over_fixed_rank_tensors<double>(test_full_size_set3<double>, 3);
}

//
// COMPLEX SPECIALIZATIONS
//

TEST(SliceSetTest, SliceCTensor1DFullSet) {
  test_over_fixed_rank_tensors<cdouble>(test_full_size_set1<cdouble>, 1);
}

TEST(SliceSetTest, SliceCTensor2DFullSet) {
  test_over_fixed_rank_tensors<cdouble>(test_full_size_set2<cdouble>, 2);
}

TEST(SliceSetTest, SliceCTensor3DFullSet) {
  test_over_fixed_rank_tensors<cdouble>(test_full_size_set3<cdouble>, 3);
}

//////////////////////////////////////////////////////////////////////
// FULL RANGE ASSIGNMENT
//

template <typename elt_t>
void test_full_size_set_number1(Tensor<elt_t> &P) {
  Tensor<elt_t> t = P(_);
  t.fill_with(number_one<elt_t>());
  if (t.size()) ASSERT_FALSE(all_equal(t, P));
  P.at(_) = number_one<elt_t>();
  ASSERT_TRUE(all_equal(P, t));
}

template <typename elt_t>
void test_full_size_set_number2(Tensor<elt_t> &P) {
  Tensor<elt_t> t = P(_, _);
  t.fill_with(number_one<elt_t>());
  if (t.size()) ASSERT_FALSE(all_equal(t, P));
  P.at(_, _) = number_one<elt_t>();
  ASSERT_TRUE(all_equal(P, t));
}

template <typename elt_t>
void test_full_size_set_number3(Tensor<elt_t> &P) {
  Tensor<elt_t> t = P(_, _, _);
  t.fill_with(number_one<elt_t>());
  if (t.size()) ASSERT_FALSE(all_equal(t, P));
  P.at(_, _, _) = number_one<elt_t>();
  ASSERT_TRUE(all_equal(P, t));
}

//
// REAL SPECIALIZATIONS
//

TEST(SliceSetTest, SliceRTensor1DFullSetNumber) {
  test_over_fixed_rank_tensors<double>(test_full_size_set_number1<double>, 1);
}

TEST(SliceSetTest, SliceRTensor2DFullSetNumber) {
  test_over_fixed_rank_tensors<double>(test_full_size_set_number2<double>, 2);
}

TEST(SliceSetTest, SliceRTensor3DFullSetNumber) {
  test_over_fixed_rank_tensors<double>(test_full_size_set_number3<double>, 3);
}

//
// COMPLEX SPECIALIZATIONS
//

TEST(SliceSetTest, SliceCTensor1DFullSetNumber) {
  test_over_fixed_rank_tensors<cdouble>(test_full_size_set_number1<cdouble>, 1);
}

TEST(SliceSetTest, SliceCTensor2DFullSetNumber) {
  test_over_fixed_rank_tensors<cdouble>(test_full_size_set_number2<cdouble>, 2);
}

TEST(SliceSetTest, SliceCTensor3DFullSetNumber) {
  test_over_fixed_rank_tensors<cdouble>(test_full_size_set_number3<cdouble>, 3);
}

}  // namespace tensor_test
