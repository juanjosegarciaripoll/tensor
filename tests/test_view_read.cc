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

#include "test_view_common.cc"

//////////////////////////////////////////////////////////////////////
// MANUALLY EXTRACT RANGES FROM A TENSOR USING LOOPS
//

template <typename elt_t>
Tensor<elt_t> slow_range1(const Tensor<elt_t> &P, index_t i0, index_t i2,
                          index_t i1) {
  auto t = Tensor<elt_t>::empty((i2 - i0) / i1 + 1);
  for (index_t j = i0, x = 0; j <= i2; j += i1, x++) {
    t.at(x) = P(j);
  }
  return t;
}

template <typename elt_t>
Tensor<elt_t> slow_range2(const Tensor<elt_t> &P, index_t i0, index_t i2,
                          index_t i1, index_t j0, index_t j2, index_t j1) {
  auto t = Tensor<elt_t>::empty((i2 - i0) / i1 + 1, (j2 - j0) / j1 + 1);
  for (index_t i = i0, x = 0; i <= i2; i += i1, x++) {
    for (index_t j = j0, y = 0; j <= j2; j += j1, y++) {
      t.at(x, y) = P(i, j);
    }
  }
  return t;
}

template <typename elt_t>
Tensor<elt_t> slow_range3(const Tensor<elt_t> &P, index_t i0, index_t i2,
                          index_t i1, index_t j0, index_t j2, index_t j1,
                          index_t k0, index_t k2, index_t k1) {
  auto t = Tensor<elt_t>::empty((i2 - i0) / i1 + 1, (j2 - j0) / j1 + 1,
                                (k2 - k0) / k1 + 1);
  for (index_t i = i0, x = 0; i <= i2; i += i1, x++) {
    for (index_t j = j0, y = 0; j <= j2; j += j1, y++) {
      for (index_t k = k0, z = 0; k <= k2; k += k1, z++) {
        t.at(x, y, z) = P(i, j, k);
      }
    }
  }
  return t;
}

template <typename elt_t>
Tensor<elt_t> slow_range4(const Tensor<elt_t> &P, index_t i0, index_t i2,
                          index_t i1, index_t j0, index_t j2, index_t j1,
                          index_t k0, index_t k2, index_t k1, index_t l0,
                          index_t l2, index_t l1) {
  auto t = Tensor<elt_t>((i2 - i0) / i1 + 1, (j2 - j0) / j1 + 1,
                         (k2 - k0) / k1 + 1, (l2 - l0) / l1 + 1);
  for (index_t i = i0, x = 0; i <= i2; i += i1, x++) {
    for (index_t j = j0, y = 0; j <= j2; j += j1, y++) {
      for (index_t k = k0, z = 0; k <= k2; k += k1, z++) {
        for (index_t l = l0, w = 0; l <= l2; l += l1, z++) {
          t.at(x, y, z, w) = P(i, j, k, l);
        }
      }
    }
  }
  return t;
}

//////////////////////////////////////////////////////////////////////
// FULL RANGE EXTRACTION
//

template <typename elt_t>
void test_full_size_range1(Tensor<elt_t> &P) {
  Tensor<elt_t> Paux = P;
  Tensor<elt_t> t = P(_);
  ASSERT_TRUE(all_equal(P, t));
  unchanged(P, Paux);
}

template <typename elt_t>
void test_full_size_range2(Tensor<elt_t> &P) {
  Tensor<elt_t> Paux = P;
  Tensor<elt_t> t = P(_, _);
  ASSERT_TRUE(all_equal(P, t));
  unchanged(P, Paux);
}

template <typename elt_t>
void test_full_size_range3(Tensor<elt_t> &P) {
  Tensor<elt_t> Paux = P;
  Tensor<elt_t> t = P(_, _, _);
  ASSERT_TRUE(all_equal(P, t));
  unchanged(P, Paux);
}

//////////////////////////////////////////////////////////////////////
// UNIT SIZE RANGE EXTRACTION
//

template <typename elt_t>
void test_extract_unit_size1(Tensor<elt_t> &P, index_t i) {
  SCOPED_TRACE("extract unit range 1D");
  Tensor<elt_t> Paux = P;
  {
    Tensor<elt_t> t = P(range(i));
    ASSERT_EQ(t.rank(), 1);
    ASSERT_EQ(t.size(), 1);
    ASSERT_EQ(t[0], P(i));
  }
  {
    Tensor<elt_t> t = P(range(i, i));
    ASSERT_EQ(t.rank(), 1);
    ASSERT_EQ(t.size(), 1);
    ASSERT_EQ(t[0], P(i));
  }
  {
    Tensor<elt_t> t = P(range(i, i, 1));
    ASSERT_EQ(t.rank(), 1);
    ASSERT_EQ(t.size(), 1);
    ASSERT_EQ(t[0], P(i));
  }
  if (i + 1 < P.dimension(0)) {
    Tensor<elt_t> t = P(range(i, i + 1, 2));
    ASSERT_EQ(t.rank(), 1);
    ASSERT_EQ(t.size(), 1);
    ASSERT_EQ(t[0], P(i));
  }
  {
    Indices ndx(1);
    ndx.at(0) = i;
    Tensor<elt_t> t = P(range(ndx));
    ASSERT_EQ(t.rank(), 1);
    ASSERT_EQ(t.size(), 1);
    ASSERT_EQ(t[0], P(i));
  }
  unchanged(P, Paux);
}

template <typename elt_t>
void test_extract_unit_size2(Tensor<elt_t> &P, index_t i, index_t j) {
  SCOPED_TRACE("extract unit range 2D");
  Tensor<elt_t> Paux = P;
  {
    Tensor<elt_t> t = P(range(i), range(j));
#ifdef TENSOR_RANGE_SQUEEZE
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1}));
#else
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1}));
#endif
    ASSERT_EQ(t.size(), 1);
    ASSERT_EQ(t[0], P(i, j));
  }
  {
    Tensor<elt_t> t = P(range(i), range(j, j));
#ifdef TENSOR_RANGE_SQUEEZE
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1}));
#else
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1}));
#endif
    ASSERT_EQ(t.size(), 1);
    ASSERT_EQ(t[0], P(i, j));
  }
  {
    Tensor<elt_t> t = P(range(i, i), range(j, j));
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1}));
    ASSERT_EQ(t[0], P(i, j));
  }
  {
    Tensor<elt_t> t = P(range(i, i), range(j));
#ifdef TENSOR_RANGE_SQUEEZE
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1}));
#else
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1}));
#endif
    ASSERT_EQ(t.size(), 1);
    ASSERT_EQ(t[0], P(i, j));
  }
  if (i + 1 < P.dimension(0)) {
    Tensor<elt_t> t = P(range(i, i + 1, 2), range(j));
#ifdef TENSOR_RANGE_SQUEEZE
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1}));
#else
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1}));
#endif
    ASSERT_EQ(t.size(), 1);
    ASSERT_EQ(t[0], P(i, j));
  }
  {
    Tensor<elt_t> t = P(range2(i, i + 1, 2), range(j));
#ifdef TENSOR_RANGE_SQUEEZE
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1}));
#else
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1}));
#endif
    ASSERT_EQ(t.size(), 1);
    ASSERT_EQ(t[0], P(i, j));
  }
  {
    Tensor<elt_t> t = P(range(i), range(j, j));
#ifdef TENSOR_RANGE_SQUEEZE
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1}));
#else
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1}));
#endif
    ASSERT_EQ(t.size(), 1);
    ASSERT_EQ(t[0], P(i, j));
  }
  unchanged(P, Paux);
}

template <typename elt_t>
void test_extract_unit_size3(Tensor<elt_t> &P, index_t i, index_t j,
                             index_t k) {
  SCOPED_TRACE("extract unit range 3D");
  Tensor<elt_t> Paux = P;
  {
    Tensor<elt_t> t = P(range(i), range(j), range(k));
#ifdef TENSOR_RANGE_SQUEEZE
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1}));
#else
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1, 1}));
#endif
    ASSERT_EQ(1, t.size());
    ASSERT_EQ(t[0], P(i, j, k));
  }
  {
    Tensor<elt_t> t = P(range(i), range(j, j), range(k));
#ifdef TENSOR_RANGE_SQUEEZE
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1}));
#else
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1, 1}));
#endif
    ASSERT_EQ(1, t.size());
    ASSERT_EQ(t[0], P(i, j, k));
  }
  {
    Tensor<elt_t> t = P(range(i, i), range(j, j), range(k));
#ifdef TENSOR_RANGE_SQUEEZE
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1}));
#else
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1, 1}));
#endif
    ASSERT_EQ(1, t.size());
    ASSERT_EQ(t[0], P(i, j, k));
  }
  {
    Tensor<elt_t> t = P(range(i, i), range(j), range(k));
#ifdef TENSOR_RANGE_SQUEEZE
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1}));
#else
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1, 1}));
#endif
    ASSERT_EQ(1, t.size());
    ASSERT_EQ(t[0], P(i, j, k));
  }
  if (i + 1 < P.dimension(0)) {
    Tensor<elt_t> t = P(range(i, i + 1, 2), range(j), range(k));
#ifdef TENSOR_RANGE_SQUEEZE
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1}));
#else
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1, 1}));
#endif
    ASSERT_EQ(1, t.size());
    ASSERT_EQ(t[0], P(i, j, k));
  }
  {
    Tensor<elt_t> t = P(range2(i, i + 1, 2), range(j), range(k));
#ifdef TENSOR_RANGE_SQUEEZE
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1}));
#else
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1, 1}));
#endif
    ASSERT_EQ(1, t.size());
    ASSERT_EQ(t[0], P(i, j, k));
  }
  {
    Tensor<elt_t> t = P(range(i), range(j, j), range(k));
#ifdef TENSOR_RANGE_SQUEEZE
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1}));
#else
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1, 1}));
#endif
    ASSERT_EQ(1, t.size());
    ASSERT_EQ(t[0], P(i, j, k));
  }
  {
    Tensor<elt_t> t = P(range(i), range(j, j), range(k, k));
#ifdef TENSOR_RANGE_SQUEEZE
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1}));
#else
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1, 1}));
#endif
    ASSERT_EQ(1, t.size());
    ASSERT_EQ(t[0], P(i, j, k));
  }
  {
    Tensor<elt_t> t = P(range(i), range(j, j), range2(k, k + 1, 2));
#ifdef TENSOR_RANGE_SQUEEZE
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1}));
#else
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{1, 1, 1}));
#endif
    ASSERT_EQ(1, t.size());
    ASSERT_EQ(t[0], P(i, j, k));
  }
  unchanged(P, Paux);
}

//////////////////////////////////////////////////////////////////////
// STEPWISE RANGE EXTRACTION
//

template <typename elt_t>
void test_view_extract1(Tensor<elt_t> &P, index_t i0, index_t i2, index_t i1) {
  SCOPED_TRACE("extract view 1D");
  Tensor<elt_t> Paux = P;

  Tensor<elt_t> t1 = slow_range1(P, i0, i2, i1);
  {
    Tensor<elt_t> t = P(range(i0, i2, i1));
    ASSERT_TRUE(all_equal(t, t1));
  }
  {
    Tensor<elt_t> t = P(range2(i0, i2, i1));
    ASSERT_TRUE(all_equal(t, t1));
  }
  if (t1.dimension(0) == 1) {
    Tensor<elt_t> t = P(range(i0));
    ASSERT_TRUE(all_equal(t, t1));
  }
  if (t1.dimension(0) == P.dimension(0)) {
    Tensor<elt_t> t = P(_);
    ASSERT_TRUE(all_equal(t, t1));
  }
  unchanged(P, Paux);
}

template <typename elt_t>
void test_view_extract(Tensor<elt_t> &P, index_t i0, index_t i2, index_t i1,
                       index_t j0, index_t j2, index_t j1) {
  SCOPED_TRACE("extract view 2D");
  Tensor<elt_t> Paux = P;
  Tensor<elt_t> t1 = slow_range2(P, i0, i2, i1, j0, j2, j1);
  {
    Tensor<elt_t> t = P(range(i0, i2, i1), range(j0, j2, j1));
    ASSERT_TRUE(all_equal(t, t1));
  }
  {
    Tensor<elt_t> t = P(range2(i0, i2, i1), range(j0, j2, j1));
    ASSERT_TRUE(all_equal(t, t1));
  }
  {
    Tensor<elt_t> t = P(range(i0, i2, i1), range2(j0, j2, j1));
    ASSERT_TRUE(all_equal(t, t1));
  }
  {
    Tensor<elt_t> t = P(range2(i0, i2, i1), range2(j0, j2, j1));
    ASSERT_TRUE(all_equal(t, t1));
  }
  if (t1.dimension(0) == 1) {
    Tensor<elt_t> t = P(range(i0), range(j0, j2, j1));
    ASSERT_EQ(t.size(), t1.size());
#ifdef TENSOR_RANGE_SQUEEZE
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{t1.size()}));
    ASSERT_TRUE(all_equal(flatten(t), flatten(t1)));
#else
    ASSERT_TRUE(all_equal(t, t1));
#endif
  }
  if (t1.dimension(1) == 1) {
    Tensor<elt_t> t = P(range(i0, i2, i1), range(j0));
#ifdef TENSOR_RANGE_SQUEEZE
    ASSERT_TRUE(all_equal(t.dimensions(), Dimensions{t1.size()}));
    ASSERT_TRUE(all_equal(flatten(t), flatten(t1)));
#else
    ASSERT_TRUE(all_equal(t, t1));
#endif
  }
  if (t1.dimension(0) == P.dimension(0)) {
    Tensor<elt_t> t = P(_, range(j0, j2, j1));
    ASSERT_TRUE(all_equal(t, t1));
  }
  if (t1.dimension(1) == P.dimension(1)) {
    Tensor<elt_t> t = P(range(i0, i2, i1), _);
    ASSERT_TRUE(all_equal(t, t1));
  }
  unchanged(P, Paux);
}

template <typename elt_t>
void test_view_extract(Tensor<elt_t> &P, index_t i0, index_t i2, index_t i1,
                       index_t j0, index_t j2, index_t j1, index_t k0,
                       index_t k2, index_t k1) {
  SCOPED_TRACE("extract view 3D");
  Tensor<elt_t> Paux = P;

  Tensor<elt_t> t1 = slow_range3(P, i0, i2, i1, j0, j2, j1, k0, k2, k1);
  Tensor<elt_t> t = P(range(i0, i2, i1), range(j0, j2, j1), range(k0, k2, k1));
  ASSERT_TRUE(all_equal(t, t1));

  unchanged(P, Paux);
}

//////////////////////////////////////////////////////////////////////
// DRIVERS FOR EXTRACTION TESTS
//

template <typename elt_t>
void test_range_extract1(Tensor<elt_t> &P) {
  test_full_size_range1(P);

  index_t d0 = P.dimension(0);
  for (index_t i = 0; i < d0; i++) {
    test_extract_unit_size1(P, i);
  }
  for (index_t i1 = 1; i1 < 4; i1++) {
    for (index_t i0 = 0; i0 < d0; i0++) {
      for (index_t i2 = i0; i2 < d0; i2++) {
        test_view_extract1(P, i0, i2, i1);
      }
    }
  }
}

template <typename elt_t>
void test_range_extract2(Tensor<elt_t> &P) {
  test_full_size_range2(P);

  index_t rows = P.dimension(0);
  index_t cols = P.dimension(1);
  for (index_t i = 0; i < rows; i++) {
    for (index_t j = 0; j < cols; j++) {
      test_extract_unit_size2(P, i, j);
    }
  }
  for (index_t i1 = 1; i1 < 4; i1++) {
    for (index_t j1 = 1; j1 < 4; j1++) {
      for (index_t i0 = 0; i0 < rows; i0++) {
        for (index_t j0 = 0; j0 < cols; j0++) {
          for (index_t i2 = i0; i2 < rows; i2++) {
            for (index_t j2 = j0; j2 < cols; j2++) {
              test_view_extract(P, i0, i2, i1, j0, j2, j1);
            }
          }
        }
      }
    }
  }
}

template <typename elt_t>
void test_range_extract3(Tensor<elt_t> &P) {
  test_full_size_range3(P);

  index_t d0 = P.dimension(0);
  index_t d1 = P.dimension(1);
  index_t d2 = P.dimension(2);
  for (index_t i = 0; i < d0; i++) {
    for (index_t j = 0; j < d1; j++) {
      for (index_t k = 0; k < d2; k++) {
        test_extract_unit_size3(P, i, j, k);
      }
    }
  }
  for (index_t i1 = 1; i1 < 3; i1++) {
    for (index_t j1 = 1; j1 < 3; j1++) {
      for (index_t k1 = 1; k1 < 3; k1++) {
        for (index_t i0 = 0; i0 < d0; i0++) {
          for (index_t j0 = 0; j0 < d1; j0++) {
            for (index_t k0 = 0; k0 < d2; k0++) {
              for (index_t i2 = i0; i2 < d0; i2++) {
                for (index_t j2 = j0; j2 < d1; j2++) {
                  for (index_t k2 = k0; k2 < d2; k2++) {
                    test_view_extract(P, i0, i2, i1, j0, j2, j1, k0, k2, k1);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

TEST(SliceTest, ImplicitIndexConversionToRange) {
  /*
   * This function compiles only when the first argument of a slicing method being
   * a range() forces the rest to be implicitely coerced to ranges.
   */
  RTensor aux{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

  RTensor aux2 = aux(range(0), Indices{0, 1, 2});
  EXPECT_ALL_EQUAL(aux2, RTensor({1.0, 2.0, 3.0}));

  RTensor aux3 = aux(_, 0);
  EXPECT_ALL_EQUAL(aux3, RTensor({1.0, 4.0}));
}

TEST(SliceTest, SliceRTensor1DExtract) {
  test_over_fixed_rank_tensors<double>(test_range_extract1<double>, 1);
}

TEST(SliceTest, SliceRTensor2DExtract) {
  test_over_fixed_rank_tensors<double>(test_range_extract2<double>, 2);
}

TEST(SliceTest, SliceRTensor3DExtract) {
  test_over_fixed_rank_tensors<double>(test_range_extract3<double>, 3, 6);
}

/////////////////////////////////////////////////////////////////////
// COMPLEX SPECIALIZATIONS
//

TEST(SliceTest, SliceCTensor1DExtract) {
  test_over_fixed_rank_tensors<double>(test_range_extract1<double>, 1);
}

TEST(SliceTest, SliceCTensor2DExtract) {
  test_over_fixed_rank_tensors<double>(test_range_extract2<double>, 2);
}

TEST(SliceTest, SliceCTensor3DExtract) {
  test_over_fixed_rank_tensors<double>(test_range_extract3<double>, 3, 6);
}

}  // namespace tensor_test
