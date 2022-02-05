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
#include <strstream>
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

static bool is_empty_range_iterator(const RangeIterator &it) {
  return (*it == 0) && (it.finished()) && !it.has_next();
}

template <typename elt_t>
std::ostream &operator<<(std::ostream &out, const std::vector<elt_t> &v) {
  const char *comma = "";
  out << "[";
  for (auto &x : v) {
    out << comma << x;
    comma = ",";
  }
  out << "]";
  return out;
}

/////////////////////////////////////////////////////////////////////
// RANGE ITERATOR OPTIMIZATIONS
//
TEST(RangeIteratorTest, OptimizesEmptyRanges01) {
  auto it = RangeIterator::begin({Range::empty(), Range(0, 3)});
  ASSERT_TRUE(is_empty_range_iterator(it));
}

TEST(RangeIteratorTest, OptimizesEmptyRanges10) {
  auto it = RangeIterator::begin({Range(0, 3), Range::empty()});
  ASSERT_TRUE(is_empty_range_iterator(it));
}

TEST(RangeIteratorTest, OptimizesEmptyRanges011) {
  auto it = RangeIterator::begin({Range::empty(), Range(0, 3), Range(0, 4)});
  ASSERT_TRUE(is_empty_range_iterator(it));
}

TEST(RangeIteratorTest, OptimizesEmptyRanges101) {
  auto it = RangeIterator::begin({Range(0, 3), Range::empty(), Range(0, 4)});
  ASSERT_TRUE(is_empty_range_iterator(it));
}

TEST(RangeIteratorTest, OptimizesEmptyRanges110) {
  auto it = RangeIterator::begin({Range(0, 5), Range(0, 3), Range::empty()});
  ASSERT_TRUE(is_empty_range_iterator(it));
}

TEST(RangeIteratorTest, OptimizesSize1) {
  /* Two size 1 ranges are combined into a single iterator. */
  auto r1 = Range(/*start*/ 1, /*end*/ 1, /*step*/ 1, /*dimension*/ 2);
  ASSERT_EQ(r1.size(), 1);
  auto r2 = Range(/*start*/ 2, /*end*/ 2, /*step*/ 1, /*dimension*/ 3);
  ASSERT_EQ(r2.size(), 1);
  auto it = RangeIterator::begin({r1, r2});
  ASSERT_FALSE(it.has_next());
  ASSERT_EQ(it.counter(), 1 + 2 * 2);
  ASSERT_EQ(it.step(), 2 * 1);
  ASSERT_EQ(it.limit(), 1 + 2 * 3);
  ASSERT_EQ(it.offset(), 0);
}

/////////////////////////////////////////////////////////////////////
// 1D RANGE ITERATORS
//

TEST(RangeIteratorTest, EmptyRangeIterator) {
  Range r = Range::empty();  // = []
  RangeIterator it(r, 1);
  ASSERT_EQ(*it, 0);
  ASSERT_TRUE(it.finished());
  ASSERT_EQ(*it, 0);
  ASSERT_EQ(it, RangeIterator::end({r}));
}

TEST(RangeIteratorTest, RangeIterator1DSize0) {
  Range r(0, -1);  // = []
  RangeIterator it(r, 1);
  ASSERT_EQ(*it, 0);
  ASSERT_TRUE(it.finished());
  ++it;
  ASSERT_EQ(*it, 0);  // We do not run past the limit
  ASSERT_EQ(it, RangeIterator::end({r}));
}

TEST(RangeIteratorTest, RangeIterator1DSize1) {
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

TEST(RangeIteratorTest, RangeIterator1DSize1Start1) {
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

TEST(RangeIteratorTest, RangeIterator1DSize2) {
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

TEST(RangeIteratorTest, RangeIterator1DSize1Step2) {
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

TEST(RangeIteratorTest, RangeIterator1DSize2Step2) {
  Range r(/*start*/ 0, /*end*/ 1, /*step*/ 2, /*dimension*/ 2);  // = [0]
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

/*
 * When any range is empty, ranges combine together to form an empty range, and
 * the resulting iterator is also empty.
 */
TEST(RangeIteratorTest, RangeIterator2DEmptyA) {
  Range r1 = Range::empty(), r2(/*start*/ 0, /*end*/ 0);  // = []
  r1.set_dimension(3);
  r2.set_dimension(3);
  RangeIterator it = RangeIterator::make_range_iterators({r1, r2});
  ASSERT_EQ(*it, 0);
  ASSERT_TRUE(it.finished());
  ASSERT_EQ(*(++it), 0);
  ASSERT_TRUE(it == RangeIterator::end({r1, r2}));
}

TEST(RangeIteratorTest, RangeIterator2DEmptyB) {
  Range r1(/*start*/ 0, /*end*/ 0), r2 = Range::empty();  // = []
  r1.set_dimension(3);
  r2.set_dimension(3);
  RangeIterator it = RangeIterator::make_range_iterators({r1, r2});
  ASSERT_EQ(*it, 0);
  ASSERT_TRUE(it.finished());
  ASSERT_EQ(*(++it), 0);  // We do not run past the limit
  ASSERT_TRUE(it == RangeIterator::end({r1, r2}));
}

TEST(RangeIteratorTest, RangeIterator2DSize1x1) {
  Range r1(/*start*/ 0, /*end*/ 0, /*step*/ 1, /*dimension*/ 1);
  Range r2(/*start*/ 0, /*end*/ 0, /*step*/ 1, /*dimension*/ 1);  // = [[0, 0]]
  RangeIterator it = RangeIterator::make_range_iterators({r1, r2});
  ASSERT_EQ(*it, 0);
  ASSERT_FALSE(it.finished());
  ++it;
  ASSERT_EQ(*it, 1);
  ASSERT_TRUE(it.finished());
  ++it;
  ASSERT_EQ(*it, 1);  // We do not run past the limit
  ASSERT_TRUE(it == RangeIterator::end({r1, r2}));
}

TEST(RangeIteratorTest, RangeIterator2DSize1x1Dim3x4) {
  Range r1(/*start*/ 0, /*end*/ 0, /*step*/ 1, /*dimension*/ 3);
  Range r2(/*start*/ 0, /*end*/ 0, /*step*/ 1, /*dimension*/ 4);
  RangeIterator it = RangeIterator::make_range_iterators({r1, r2});
  ASSERT_EQ(*it, 0);
  ASSERT_FALSE(it.finished());
  ++it;
  ASSERT_EQ(*it, 3);
  ASSERT_TRUE(it.finished());
  ++it;
  ASSERT_EQ(*it, 3);  // We do not run past the limit
  ASSERT_TRUE(it == RangeIterator::end({r1, r2}));
}

TEST(RangeIteratorTest, RangeIterator2DSize2x2Dim3x4) {
  Range r1(/*start*/ 0, /*end*/ 1, /*step*/ 1, /*dimension*/ 3);
  Range r2(/*start*/ 0, /*end*/ 1, /*step*/ 1, /*dimension*/ 4);
  RangeIterator it = RangeIterator::make_range_iterators({r1, r2});
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
  ASSERT_TRUE(it == RangeIterator::end({r1, r2}));
}

//////////////////////////////////////////////////////////////////////
// TEST RANGE ITERATION COMPARING WITH MANUALLY CRAFTED LOOPS
//
// 1) ONLY PURE RANGES
//

template <typename elt_t>
Tensor<elt_t> slow_range_test1(const Tensor<elt_t> &P, index i0, index i2,
                               index i1) {
  SimpleVector<Range> ranges{range(i0, i2, i1)};
  Dimensions dims = dimensions_from_ranges(ranges, P.dimensions());
  RangeIterator it = RangeIterator::begin(ranges);
  index count = 0;
  for (index i = i0; i <= i2; i += i1, ++it, ++count) {
    EXPECT_EQ(i, *it);
    if (i != *it) {
      std::stringstream out;
      out << "Mismatch in range over ranges:\n"
          << ranges << '\n'
          << "Mismatch: " << i << " != " << *it << '\n';
      throw std::overflow_error(out.str());
    }
  }
  EXPECT_EQ(count, dims.total_size());
  EXPECT_TRUE(it.finished());
  return P;
}

TEST(RangeIteratorTest, Test1D) {
  test_over_fixed_rank_tensors<double>(
      [](const Tensor<double> &P) {
        index d0 = P.dimension(0);
        for (index i1 = 1; i1 < 4; i1++) {
          for (index i0 = 0; i0 < d0; i0++) {
            for (index i2 = i0; i2 < d0; i2++) {
              slow_range_test1(P, i0, i2, i1);
            }
          }
        }
      },
      1);
}

template <typename elt_t>
Tensor<elt_t> slow_range_test2(const Tensor<elt_t> &P, index i0, index i2,
                               index i1, index j0, index j2, index j1) {
  SimpleVector<Range> ranges{range(i0, i2, i1), range(j0, j2, j1)};
  Dimensions dims = dimensions_from_ranges(ranges, P.dimensions());
  RangeIterator it = RangeIterator::begin(ranges);
  index count = 0;
  for (index j = j0; j <= j2; j += j1) {
    for (index i = i0; i <= i2; i += i1) {
      index pos = i + j * P.dimension(0);
      EXPECT_EQ(pos, *it);
      if (pos != *it) {
        std::stringstream out;
        out << "Mismatch in range over ranges:\n"
            << ranges << '\n'
            << "Mismatch: " << pos << " != " << *it << '\n';
        throw std::overflow_error(out.str());
      }
      ++it;
      ++count;
    }
  }
  EXPECT_EQ(count, dims.total_size());
  EXPECT_TRUE(it.finished());
  return P;
}

TEST(RangeIteratorTest, Test2D) {
  test_over_fixed_rank_tensors<double>(
      [](const Tensor<double> &P) {
        index rows = P.dimension(0);
        index cols = P.dimension(1);
        for (index i1 = 1; i1 < 4; i1++) {
          for (index j1 = 1; j1 < 4; j1++) {
            for (index i0 = 0; i0 < rows; i0++) {
              for (index j0 = 0; j0 < cols; j0++) {
                for (index i2 = i0; i2 < rows; i2++) {
                  for (index j2 = j0; j2 < cols; j2++) {
                    slow_range_test2(P, i0, i2, i1, j0, j2, j1);
                  }
                }
              }
            }
          }
        }
      },
      2);
}

#if 0
template <typename elt_t>
Tensor<elt_t> slow_range_test3(const Tensor<elt_t> &P, index i0, index i2,
                               index i1, index j0, index j2, index j1, index k0,
                               index k2, index k1) {
  SimpleVector<Range> ranges{range(i0, i2, i1), range(j0, j2, j1),
                             range(k0, k2, k1)};
  Dimensions dims = dimensions_from_ranges(ranges, P.dimensions());
  RangeIterator it = RangeIterator::begin(ranges);
  index count = 0;
  for (index k = k0; k <= k2; k += k1) {
    for (index j = j0; j <= j2; j += j1) {
      for (index i = i0; i <= i2; i += i1, ++it, ++count) {
        index pos = i + P.dimension(0) * (j + k * P.dimension(1));
        EXPECT_EQ(pos, *it);
        if (pos != *it) {
          std::stringstream out;
          out << "Mismatch in range over ranges:\n"
              << ranges << '\n'
              << "Tensor: \n"
              << P << '\n'
              << "Indices: " << i << ',' << j << ',' << k << '\n'
              << "Mismatch: " << pos << " != " << *it << '\n';
          throw std::overflow_error(out.str());
        }
      }
    }
  }
  EXPECT_EQ(count, dims.total_size());
  EXPECT_TRUE(it.finished());
  return P;
}

TEST(RangeIteratorTest, Test3D) {
  test_over_fixed_rank_tensors<double>(
      [](const Tensor<double> &P) {
        index d0 = P.dimension(0);
        index d1 = P.dimension(1);
        index d2 = P.dimension(2);
        for (index i1 = 1; i1 < 3; i1++) {
          for (index j1 = 1; j1 < 3; j1++) {
            for (index k1 = 1; k1 < 3; k1++) {
              for (index i0 = 0; i0 < d0; i0++) {
                for (index j0 = 0; j0 < d1; j0++) {
                  for (index k0 = 0; k0 < d2; k0++) {
                    for (index i2 = i0; i2 < d0; i2++) {
                      for (index j2 = j0; j2 < d1; j2++) {
                        for (index k2 = k0; k2 < d2; k2++) {
                          slow_range_test3(P, i0, i2, i1, j0, j2, j1, k0, k2,
                                           k1);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      },
      3);
}
#endif

//
// 1) WITH INDICES
//

Indices random_indices(index size) {
  return which(RTensor::random(size) < 0.5);
}

template <typename elt_t>
Tensor<elt_t> slow_index_range_test1(const Tensor<elt_t> &P) {
  for (index attempts = 10; attempts; --attempts) {
    const Indices ndx = random_indices(P.size());
    SimpleVector<Range> ranges{range(ndx)};
    Dimensions dims = dimensions_from_ranges(ranges, P.dimensions());
    RangeIterator it = RangeIterator::begin(ranges);
    for (index i = 0; i < ndx.size(); ++i, ++it) {
      EXPECT_EQ(ndx[i], *it);
      if (ndx[i] != *it) {
        std::stringstream out;
        out << "Mismatch in range over ranges:\n"
            << ranges << '\n'
            << "Mismatch: " << ndx[i] << " != " << *it << '\n';
        throw std::overflow_error(out.str());
      }
    }
    EXPECT_TRUE(it.finished());
  }
  return P;
}

TEST(RangeIteratorTest, Test1DIndices) {
  test_over_fixed_rank_tensors<double>(slow_index_range_test1<double>, 1);
  test_over_fixed_rank_tensors<double>(slow_index_range_test1<double>, 2);
  test_over_fixed_rank_tensors<double>(slow_index_range_test1<double>, 3);
}

template <typename elt_t>
Tensor<elt_t> slow_index_range_test2a(const Tensor<elt_t> &P) {
  for (index attempts = 10; attempts; --attempts) {
    const Indices ndx = random_indices(P.dimension(0));
    SimpleVector<Range> ranges{range(ndx), range()};
    Dimensions dims = dimensions_from_ranges(ranges, P.dimensions());
    RangeIterator it = RangeIterator::begin(ranges);
    for (index j = 0; j < P.dimension(1); ++j) {
      for (index i = 0; i < ndx.size(); ++i, ++it) {
        index pos = ndx[i] + j * P.dimension(0);
        EXPECT_EQ(pos, *it);
        if (pos != *it) {
          std::stringstream out;
          out << "Mismatch in range over ranges:\n"
              << ranges << '\n'
              << "Indices:" << ndx << '\n'
              << "Coordinates: (" << ndx[i] << "," << j << ")\n"
              << "Mismatch: " << pos << " != " << *it << '\n';
          throw std::overflow_error(out.str());
        }
      }
    }
    EXPECT_TRUE(it.finished());
  }
  return P;
}

template <typename elt_t>
Tensor<elt_t> slow_index_range_test2b(const Tensor<elt_t> &P) {
  for (index attempts = 10; attempts; --attempts) {
    const Indices ndx = random_indices(P.dimension(1));
    SimpleVector<Range> ranges{range(), range(ndx)};
    Dimensions dims = dimensions_from_ranges(ranges, P.dimensions());
    RangeIterator it = RangeIterator::begin(ranges);
    for (index j = 0; j < ndx.size(); ++j) {
      for (index i = 0; i < P.dimension(0); ++i, ++it) {
        index pos = i + ndx[j] * P.dimension(0);
        EXPECT_EQ(pos, *it);
        if (pos != *it) {
          std::stringstream out;
          out << "Mismatch in range over ranges:\n"
              << ranges << '\n'
              << "Indices:" << ndx << '\n'
              << "Coordinates: (" << i << "," << ndx[j] << ")\n"
              << "Mismatch: " << pos << " != " << *it << '\n';
          throw std::overflow_error(out.str());
        }
      }
    }
    EXPECT_TRUE(it.finished());
  }
  return P;
}

TEST(RangeIteratorTest, Test2DIndices) {
  test_over_fixed_rank_tensors<double>(slow_index_range_test2a<double>, 2);
  test_over_fixed_rank_tensors<double>(slow_index_range_test2b<double>, 2);
}

}  // namespace tensor_test