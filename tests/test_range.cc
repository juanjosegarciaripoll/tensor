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

/////////////////////////////////////////////////////////////////////
// RANGES
//

TEST(RangeTest, EmptyRange) {
  Range r = Range::empty();
  ASSERT_EQ(r.first(), -1);
  ASSERT_EQ(r.last(), -2);
  ASSERT_EQ(r.step(), 1);
  ASSERT_EQ(r.dimension(), -1);
  ASSERT_EQ(r.size(), 0);
}

TEST(RangeTest, CannotChangeDimension) {
  Range r = Range::full();
  ASSERT_NO_THROW(r.set_dimension(4));
  ASSERT_THROW_DEBUG(r.set_dimension(50), ::tensor::invalid_assertion);
}

TEST(RangeTest, Range1D) {
  Range r(0, 4, 1);
  ASSERT_EQ(r.first(), 0);
  ASSERT_EQ(r.last(), 4);
  ASSERT_EQ(r.step(), 1);
  ASSERT_EQ(r.dimension(), -1);
  ASSERT_EQ(r.size(), 5);
}

TEST(RangeTest, RangeFull) {
  Range r = Range::full(0);
  ASSERT_EQ(r.first(), 0);
  ASSERT_EQ(r.last(), -1);
  ASSERT_EQ(r.step(), 1);
  r.set_dimension(3);
  ASSERT_EQ(r.last(), 2);
  ASSERT_EQ(r.size(), 3);
}

TEST(RangeTest, RangeFullWithStart) {
  Range r = Range::full(1);
  ASSERT_EQ(r.first(), 1);
  ASSERT_EQ(r.last(), -1);
  ASSERT_EQ(r.step(), 1);
  r.set_dimension(3);
  ASSERT_EQ(r.last(), 2);
  ASSERT_EQ(r.size(), 2);
}

TEST(RangeTest, RangeEmptyIndices) {
  Range r(Indices{});
  Range empty = Range::empty();
  ASSERT_FALSE(r.has_indices());
  ASSERT_EQ(r.first(), empty.first());
  ASSERT_EQ(r.last(), empty.last());
  ASSERT_EQ(r.step(), empty.step());
  ASSERT_EQ(r.size(), 0);
  ASSERT_NO_THROW(r.set_dimension(3));
  ASSERT_EQ(r.size(), 0);
}

TEST(RangeTest, RangeIndicesSize1) {
  Range r(Indices{1});
  ASSERT_FALSE(r.has_indices());
  ASSERT_EQ(r.first(), 1);
  ASSERT_EQ(r.last(), 1);
  ASSERT_EQ(r.step(), 1);
  ASSERT_EQ(r.size(), 1);
  ASSERT_NO_THROW(r.set_dimension(3));
  ASSERT_EQ(r.size(), 1);
}

TEST(RangeTest, RangeIndicesSize2) {
  Range r(Indices{1, 3});
  ASSERT_FALSE(r.has_indices());
  ASSERT_EQ(r.first(), 1);
  ASSERT_EQ(r.last(), 3);
  ASSERT_EQ(r.step(), 2);
  ASSERT_EQ(r.size(), 2);
  ASSERT_NO_THROW(r.set_dimension(4));
  ASSERT_EQ(r.size(), 2);
}

TEST(RangeTest, RangeIndices) {
  Range r(Indices{0, 1, 3});
  ASSERT_EQ(r.first(), 0);
  ASSERT_EQ(r.last(), 2);
  ASSERT_EQ(r.step(), 1);
  ASSERT_EQ(r.size(), 3);
  Range r2 = r;
  ASSERT_THROW_DEBUG(r2.set_dimension(2), ::tensor::out_of_bounds_index);
  Range r3 = r;
  ASSERT_NO_THROW(r3.set_dimension(4));
}

/////////////////////////////////////////////////////////////////
// NEGATIVE RANGES
//

TEST(RangeTest, RangeNegativeEnd1A) {
  Range r(0, -1);
  r.set_dimension(0);
  ASSERT_EQ(r.first(), 0);
  ASSERT_EQ(r.last(), -1);
  ASSERT_EQ(r.size(), 0);
}

TEST(RangeTest, RangeNegativeEnd1B) {
  Range r(0, -1);
  r.set_dimension(3);
  ASSERT_EQ(r.first(), 0);
  ASSERT_EQ(r.last(), 2);
  ASSERT_EQ(r.size(), 3);
}

TEST(RangeTest, RangeNegativeEnd1C) {
  Range r(0, -2);
  r.set_dimension(3);
  ASSERT_EQ(r.first(), 0);
  ASSERT_EQ(r.last(), 1);
  ASSERT_EQ(r.size(), 2);
}

TEST(RangeTest, RangeNegativeEnd1D) {
  Range r(0, -3);
  r.set_dimension(3);
  ASSERT_EQ(r.first(), 0);
  ASSERT_EQ(r.last(), 0);
  ASSERT_EQ(r.size(), 1);
}

TEST(RangeTest, RangeNegativeEnd1E) {
  Range r(0, -4);
  r.set_dimension(3);
  ASSERT_EQ(r.first(), 0);
  ASSERT_EQ(r.last(), -1);
  ASSERT_EQ(r.size(), 0);
}

TEST(RangeTest, RangeNegativeBeginning1A) {
  Range r(-1, -1);
  r.set_dimension(3);
  ASSERT_EQ(r.first(), 2);
  ASSERT_EQ(r.last(), 2);
  ASSERT_EQ(r.size(), 1);
}

TEST(RangeTest, RangeNegativeBeginning1B) {
  Range r(-2, -1);
  r.set_dimension(3);
  ASSERT_EQ(r.first(), 1);
  ASSERT_EQ(r.last(), 2);
  ASSERT_EQ(r.size(), 2);
}

TEST(RangeTest, RangeNegativeBeginning1C) {
  Range r(-3, -1);
  r.set_dimension(3);
  ASSERT_EQ(r.first(), 0);
  ASSERT_EQ(r.last(), 2);
  ASSERT_EQ(r.size(), 3);
}

TEST(RangeTest, RangeNegativeBeginning1D) {
  Range r(-4, -1);
  ASSERT_THROW_DEBUG(r.set_dimension(3), out_of_bounds_index);
}

TEST(RangeTest, RangeInvertedIndices) {
  Range r(Indices{1, 0});
  Range transformed(1, 0, -1);
  ASSERT_EQ(r.first(), transformed.first());
  ASSERT_EQ(r.last(), transformed.last());
  ASSERT_EQ(r.step(), transformed.step());
}

/////////////////////////////////////////////////////////////////
// RANGE COMBINATIONS
//

TEST(RangeTest, CombineSize1Size3) {
  auto r1 = Range(/*first*/ 1, /*end*/ 1, /*step*/ 1, /*dimension*/ 4);
  ASSERT_EQ(r1.size(), 1);
  auto r2 = Range(/*first*/ 1, /*end*/ 3, /*step*/ 1, /*dimension*/ 5);
  ASSERT_EQ(r2.size(), 3);
  ASSERT_TRUE(r1.maybe_combine(r2));
  ASSERT_EQ(r1.size(), 3);
  ASSERT_EQ(r1.first(), 1 + 4 * 1);
  ASSERT_EQ(r1.step(), 4 * 1);
}

TEST(RangeTest, CombineSize3Size0) {
  auto r1 = Range(/*first*/ 1, /*end*/ 3, /*step*/ 1, /*dimension*/ 4);
  ASSERT_EQ(r1.size(), 3);
  auto r2 = Range(/*first*/ 2, /*end*/ 1, /*step*/ 1, /*dimension*/ 5);
  ASSERT_EQ(r2.size(), 0);
  ASSERT_TRUE(r1.maybe_combine(r2));
  ASSERT_EQ(r1.size(), 0);
  ASSERT_EQ(r1.first(), 0);
  ASSERT_EQ(r1.step(), 1);
}

TEST(RangeTest, CombineSize0Size3) {
  auto r1 = Range(/*first*/ 2, /*end*/ 1, /*step*/ 1, /*dimension*/ 4);
  ASSERT_EQ(r1.size(), 0);
  auto r2 = Range(/*first*/ 1, /*end*/ 3, /*step*/ 1, /*dimension*/ 5);
  ASSERT_EQ(r2.size(), 3);
  ASSERT_TRUE(r1.maybe_combine(r2));
  ASSERT_EQ(r1.size(), 0);
  ASSERT_EQ(r1.first(), 0);
  ASSERT_EQ(r1.step(), 1);
}

TEST(RangeTest, CombineSize3Size1) {
  auto r1 = Range(/*first*/ 1, /*end*/ 3, /*step*/ 1, /*dimension*/ 4);
  ASSERT_EQ(r1.size(), 3);
  auto r2 = Range(/*first*/ 2, /*end*/ 2, /*step*/ 1, /*dimension*/ 5);
  ASSERT_EQ(r2.size(), 1);
  ASSERT_TRUE(r1.maybe_combine(r2));
  ASSERT_EQ(r1.size(), 3);
  ASSERT_EQ(r1.first(), 1 + 4 * 2);
  ASSERT_EQ(r1.step(), 1);
}

TEST(RangeTest, CombineIndices3Size0) {
  auto r1 = Range(Indices{0, 1, 3});
  ASSERT_EQ(r1.size(), 3);
  auto r2 = Range::empty();
  ASSERT_EQ(r2.size(), 0);
  ASSERT_TRUE(r1.maybe_combine(r2));
  ASSERT_EQ(r1.size(), 0);
  ASSERT_EQ(r1.first(), 0);
  ASSERT_EQ(r1.step(), 1);
}

TEST(RangeTest, CombineSize0Indices3) {
  auto r1 = Range::empty();
  ASSERT_EQ(r1.size(), 0);
  auto r2 = Range(Indices{0, 1, 3});
  ASSERT_EQ(r2.size(), 3);
  ASSERT_TRUE(r1.maybe_combine(r2));
  ASSERT_EQ(r1.size(), 0);
  ASSERT_EQ(r1.first(), 0);
  ASSERT_EQ(r1.step(), 1);
}

}  // namespace tensor_test
