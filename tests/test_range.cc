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
#include <tensor/io.h>

namespace tensor_test {

using namespace tensor;
using tensor::index;

static bool is_empty_range(const Range &r) {
  return r.start() == 0 && r.limit() == 0 && r.step() == 1 && r.size() == 0 &&
         !r.has_indices();
}

/////////////////////////////////////////////////////////////////////
// RANGES
//

TEST(RangeTest, EmptyRange) {
  Range r = Range::empty();
  ASSERT_EQ(r.start(), 0);
  ASSERT_EQ(r.limit(), 0);
  ASSERT_EQ(r.step(), 1);
  ASSERT_EQ(r.dimension(), -1);
  ASSERT_EQ(r.size(), 0);
}

TEST(RangeTest, Range1D) {
  Range r(0, 4, 1);
  ASSERT_EQ(r.start(), 0);
  ASSERT_EQ(r.limit(), 5);
  ASSERT_EQ(r.step(), 1);
  ASSERT_EQ(r.dimension(), -1);
  ASSERT_THROW(r.set_dimension(3), std::out_of_range);
  ASSERT_EQ(r.size(), 5);
}

TEST(RangeTest, RangeFull) {
  Range r = Range::full(0);
  ASSERT_EQ(r.start(), 0);
  ASSERT_EQ(r.limit(), -1);
  ASSERT_EQ(r.step(), 1);
  r.set_dimension(3);
  ASSERT_EQ(r.limit(), 3);
  ASSERT_EQ(r.size(), 3);
}

TEST(RangeTest, RangeFullWithStart) {
  Range r = Range::full(1);
  ASSERT_EQ(r.start(), 1);
  ASSERT_EQ(r.limit(), -1);
  ASSERT_EQ(r.step(), 1);
  r.set_dimension(3);
  ASSERT_EQ(r.limit(), 3);
  ASSERT_EQ(r.size(), 2);
}

TEST(RangeTest, RangeEmptyIndices) {
  Range r(Indices{});
  ASSERT_FALSE(r.has_indices());
  ASSERT_EQ(r.start(), 0);
  ASSERT_EQ(r.limit(), 0);
  ASSERT_EQ(r.step(), 1);
  ASSERT_EQ(r.size(), 0);
  ASSERT_NO_THROW(r.set_dimension(3));
  ASSERT_EQ(r.size(), 0);
}

TEST(RangeTest, RangeIndicesSize1) {
  Range r(Indices{1});
  ASSERT_FALSE(r.has_indices());
  ASSERT_EQ(r.start(), 1);
  ASSERT_EQ(r.limit(), 2);
  ASSERT_EQ(r.step(), 1);
  ASSERT_EQ(r.size(), 1);
  ASSERT_NO_THROW(r.set_dimension(3));
  ASSERT_EQ(r.size(), 1);
  ASSERT_THROW(r.set_dimension(5), std::invalid_argument);
}

TEST(RangeTest, RangeIndicesSize2) {
  Range r(Indices{1, 3});
  ASSERT_FALSE(r.has_indices());
  ASSERT_EQ(r.start(), 1);
  ASSERT_EQ(r.limit(), 4);
  ASSERT_EQ(r.step(), 2);
  ASSERT_EQ(r.size(), 2);
  ASSERT_NO_THROW(r.set_dimension(4));
  ASSERT_EQ(r.size(), 2);
  ASSERT_THROW(r.set_dimension(5), std::invalid_argument);
}

TEST(RangeTest, RangeIndices) {
  Range r(Indices{0, 1, 3});
  ASSERT_EQ(r.start(), 0);
  ASSERT_EQ(r.limit(), 3);
  ASSERT_EQ(r.step(), 1);
  ASSERT_EQ(r.size(), 3);
  ASSERT_EQ(r.size(), 3);
  ASSERT_THROW(r.set_dimension(2), std::out_of_range);
  ASSERT_NO_THROW(r.set_dimension(4));
  ASSERT_THROW(r.set_dimension(5), std::invalid_argument);
}

/////////////////////////////////////////////////////////////////
// RANGE COMBINATIONS
//

TEST(RangeTest, CombineSize1Size3) {
  auto r1 = Range(/*start*/ 1, /*end*/ 1, /*step*/ 1, /*dimension*/ 4);
  ASSERT_EQ(r1.size(), 1);
  auto r2 = Range(/*start*/ 1, /*end*/ 3, /*step*/ 1, /*dimension*/ 5);
  ASSERT_EQ(r2.size(), 3);
  ASSERT_TRUE(r1.maybe_combine(r2));
  ASSERT_EQ(r1.size(), 3);
  ASSERT_EQ(r1.start(), 1 + 4 * 1);
  ASSERT_EQ(r1.limit(), 1 + 4 * 4);
  ASSERT_EQ(r1.step(), 4 * 1);
}

TEST(RangeTest, CombineSize3Size0) {
  auto r1 = Range(/*start*/ 1, /*end*/ 3, /*step*/ 1, /*dimension*/ 4);
  ASSERT_EQ(r1.size(), 3);
  auto r2 = Range(/*start*/ 2, /*end*/ 1, /*step*/ 1, /*dimension*/ 5);
  ASSERT_EQ(r2.size(), 0);
  ASSERT_TRUE(r1.maybe_combine(r2));
  ASSERT_EQ(r1.size(), 0);
  ASSERT_EQ(r1.start(), 0);
  ASSERT_EQ(r1.limit(), 0);
  ASSERT_EQ(r1.step(), 1);
}

TEST(RangeTest, CombineSize0Size3) {
  auto r1 = Range(/*start*/ 2, /*end*/ 1, /*step*/ 1, /*dimension*/ 4);
  ASSERT_EQ(r1.size(), 0);
  auto r2 = Range(/*start*/ 1, /*end*/ 3, /*step*/ 1, /*dimension*/ 5);
  ASSERT_EQ(r2.size(), 3);
  ASSERT_TRUE(r1.maybe_combine(r2));
  ASSERT_EQ(r1.size(), 0);
  ASSERT_EQ(r1.start(), 0);
  ASSERT_EQ(r1.limit(), 0);
  ASSERT_EQ(r1.step(), 1);
}

TEST(RangeTest, CombineSize3Size1) {
  auto r1 = Range(/*start*/ 1, /*end*/ 3, /*step*/ 1, /*dimension*/ 4);
  ASSERT_EQ(r1.size(), 3);
  auto r2 = Range(/*start*/ 2, /*end*/ 2, /*step*/ 1, /*dimension*/ 5);
  ASSERT_EQ(r2.size(), 1);
  ASSERT_TRUE(r1.maybe_combine(r2));
  ASSERT_EQ(r1.size(), 3);
  ASSERT_EQ(r1.start(), 1 + 4 * 2);
  ASSERT_EQ(r1.limit(), 4 + 4 * 2);
  ASSERT_EQ(r1.step(), 1);
}

TEST(RangeTest, CombineIndices3Size0) {
  auto r1 = Range(Indices{0, 1, 3});
  ASSERT_EQ(r1.size(), 3);
  auto r2 = Range::empty();
  ASSERT_EQ(r2.size(), 0);
  ASSERT_TRUE(r1.maybe_combine(r2));
  ASSERT_EQ(r1.size(), 0);
  ASSERT_EQ(r1.start(), 0);
  ASSERT_EQ(r1.limit(), 0);
  ASSERT_EQ(r1.step(), 1);
}

TEST(RangeTest, CombineSize0Indices3) {
  auto r1 = Range::empty();
  ASSERT_EQ(r1.size(), 0);
  auto r2 = Range(Indices{0, 1, 3});
  ASSERT_EQ(r2.size(), 3);
  ASSERT_TRUE(r1.maybe_combine(r2));
  ASSERT_EQ(r1.size(), 0);
  ASSERT_EQ(r1.start(), 0);
  ASSERT_EQ(r1.limit(), 0);
  ASSERT_EQ(r1.step(), 1);
}

}  // namespace tensor_test
