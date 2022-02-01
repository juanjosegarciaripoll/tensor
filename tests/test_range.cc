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

/////////////////////////////////////////////////////////////////////
// RANGES
//

TEST(RangeTest, EmptyRange) {
  Range r = Range::empty();
  ASSERT_EQ(r.start(), 0);
  ASSERT_EQ(r.limit(), 0);
  ASSERT_EQ(r.step(), 1);
  ASSERT_EQ(r.dimension(), 0);
  ASSERT_EQ(r.size(), 0);
}

TEST(RangeTest, Range1D) {
  Range r(0, 4, 1);
  ASSERT_EQ(r.start(), 0);
  ASSERT_EQ(r.limit(), 5);
  ASSERT_EQ(r.step(), 1);
  ASSERT_EQ(r.dimension(), 5);
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
  ASSERT_EQ(r.start(), 0);
  ASSERT_EQ(r.limit(), 0);
  ASSERT_EQ(r.step(), 1);
  ASSERT_EQ(r.size(), 0);
  ASSERT_NO_THROW(r.set_dimension(3));
  ASSERT_EQ(r.size(), 0);
}

TEST(RangeTest, RangeIndices) {
  Range r(Indices{0, 1, 2});
  ASSERT_EQ(r.start(), 0);
  ASSERT_EQ(r.limit(), 3);
  ASSERT_EQ(r.step(), 1);
  ASSERT_EQ(r.size(), 3);
  ASSERT_NO_THROW(r.set_dimension(3));
  ASSERT_EQ(r.size(), 3);
  ASSERT_THROW(r.set_dimension(2), std::out_of_range);
}

/////////////////////////////////////////////////////////////////////
// 1D RANGE ITERATORS
//

TEST(RangeTest, EmptyRangeIterator) {
  Range r = Range::empty();  // = []
  RangeIterator it(r, 1);
  ASSERT_TRUE(it.finished());
  ASSERT_EQ(it, RangeIterator::end({r}));
}

TEST(RangeTest, RangeIterator1DSize0) {
  Range r(0, -1);  // = []
  RangeIterator it(r, 1);
  ASSERT_EQ(*it, 0);
  ASSERT_TRUE(it.finished());
  ++it;
  ASSERT_EQ(*it, 0);  // We do not run past the limit
  ASSERT_EQ(it, RangeIterator::end({r}));
}

TEST(RangeTest, RangeIterator1DSize1) {
  Range r(/*start*/ 0, /*end*/ 0);  // = [0]
  RangeIterator it(r, 1);
  ASSERT_EQ(*it, 0);
  ASSERT_FALSE(it.finished());
  ++it;
  ASSERT_EQ(*it, 1);
  ASSERT_TRUE(it.finished());
  ++it;
  ASSERT_EQ(*it, 1);  // We do not run past the limit
  ASSERT_EQ(it, RangeIterator::end({r}));
}

TEST(RangeTest, RangeIterator1DSize1Start1) {
  Range r(/*start*/ 1, /*end*/ 1);  // = [1, 1]
  RangeIterator it(r, 1);
  ASSERT_EQ(*it, 1);
  ASSERT_FALSE(it.finished());
  ++it;
  ASSERT_EQ(*it, 2);
  ASSERT_TRUE(it.finished());
  ++it;
  ASSERT_EQ(*it, 2);  // We do not run past the limit
  ASSERT_EQ(it, RangeIterator::end({r}));
}

TEST(RangeTest, RangeIterator1DSize2) {
  Range r(/*start*/ 0, /*end*/ 1);  // = [0, 1]
  RangeIterator it(r, 1);
  ASSERT_EQ(*it, 0);
  ASSERT_FALSE(it.finished());
  ++it;
  ASSERT_EQ(*it, 1);
  ASSERT_FALSE(it.finished());
  ++it;
  ASSERT_EQ(*it, 2);
  ASSERT_TRUE(it.finished());
  ++it;
  ASSERT_EQ(*it, 2);  // We do not run past the limit
  ASSERT_EQ(it, RangeIterator::end({r}));
}

TEST(RangeTest, RangeIterator1DSize1Step2) {
  Range r(/*start*/ 0, /*end*/ 0, /*step*/ 2);  // = [0]
  RangeIterator it(r, 1);
  ASSERT_EQ(*it, 0);
  ASSERT_FALSE(it.finished());
  ++it;
  ASSERT_EQ(*it, 1);
  ASSERT_TRUE(it.finished());
  ++it;
  ASSERT_EQ(*it, 1);  // We do not run past the limit
  ASSERT_EQ(it, RangeIterator::end({r}));
}

TEST(RangeTest, RangeIterator1DSize2Step2) {
  Range r(/*start*/ 0, /*end*/ 1, /*step*/ 2);  // = [0]
  RangeIterator it(r, 1);
  ASSERT_EQ(*it, 0);
  ASSERT_FALSE(it.finished());
  ++it;
  ASSERT_EQ(*it, 2);
  ASSERT_TRUE(it.finished());
  ++it;
  ASSERT_EQ(*it, 2);  // We do not run past the limit
  ASSERT_EQ(it, RangeIterator::end({r}));
}

/////////////////////////////////////////////////////////////////////
// 2D RANGE ITERATORS
//

TEST(RangeTest, RangeIterator2DEmptyA) {
  Range r1 = Range::empty(), r2(/*start*/ 0, /*end*/ 0);  // = []
  r1.set_dimension(3);
  r2.set_dimension(3);
  RangeIterator it({r1, r2});
  ASSERT_EQ(*it, 0 + 3 * 0);
  ASSERT_TRUE(it.finished());
  ASSERT_EQ(*(++it), 0 + 3 * 0);  // We do not run past the limit
  std::cerr << it << '\n';
  std::cerr << RangeIterator::end({r1, r2}) << '\n';
  ASSERT_TRUE(it == RangeIterator::end({r1, r2}));
}

TEST(RangeTest, RangeIterator2DEmptyB) {
  Range r1(/*start*/ 0, /*end*/ 0), r2 = Range::empty();  // = []
  r1.set_dimension(3);
  r2.set_dimension(3);
  RangeIterator it({r1, r2});
  ASSERT_EQ(*it, 1 + 3 * 0);
  ASSERT_TRUE(it.finished());
  ASSERT_EQ(*(++it), 1 + 3 * 0);  // We do not run past the limit
  ASSERT_TRUE(it == RangeIterator::end({r1, r2}));
}

TEST(RangeTest, RangeIterator2DSize1) {
  Range r1(/*start*/ 0, /*end*/ 0), r2(/*start*/ 0, /*end*/ 0);  // = [[0, 0]]
  RangeIterator it({r1, r2});
  ASSERT_EQ(*it, 0);
  ASSERT_FALSE(it.finished());
  ++it;
  ASSERT_EQ(*it, 1 + 1);
  ASSERT_TRUE(it.finished());
  ++it;
  ASSERT_EQ(*it, 1 + 1);  // We do not run past the limit
  ASSERT_TRUE(it == RangeIterator::end({r1, r2}));
}

TEST(RangeTest, RangeIterator2DSize1Dim3) {
  Range r1(0, 0), r2(0, 0);
  r1.set_dimension(3);
  r2.set_dimension(3);
  RangeIterator it({r1, r2});
  ASSERT_EQ(*it, 0);
  ASSERT_FALSE(it.finished());
  ++it;
  ASSERT_EQ(*it, 1 + 3);
  ASSERT_TRUE(it.finished());
  ++it;
  ASSERT_EQ(*it, 1 + 3);  // We do not run past the limit
  ASSERT_TRUE(it == RangeIterator::end({r1, r2}));
}

TEST(RangeTest, RangeIterator2DSize2) {
  Range r1(0, 1), r2(0, 1);
  r1.set_dimension(3);
  r2.set_dimension(3);
  RangeIterator it({r1, r2});
  ASSERT_EQ(*it, 0);
  ASSERT_FALSE(it.finished());
  ASSERT_EQ(*(++it), 1);
  ASSERT_FALSE(it.finished());
  ASSERT_EQ(*(++it), 0 + 3);
  ASSERT_FALSE(it.finished());
  ASSERT_EQ(*(++it), 1 + 3);
  ASSERT_FALSE(it.finished());
  ASSERT_EQ(*(++it), 2 + 2 * 3);
  ASSERT_TRUE(it.finished());
  ASSERT_EQ(*(++it), 2 + 2 * 3);  // We do not run past the limit
  std::cerr << RangeIterator::end({r1, r2}) << '\n';
  ASSERT_TRUE(it == RangeIterator::end({r1, r2}));
}

}  // namespace tensor_test
