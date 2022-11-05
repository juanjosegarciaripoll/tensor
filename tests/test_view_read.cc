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
// MANUALLY EXTRACT RANGES FROM A TENSOR USING LOOPS
//

template <typename elt_t>
Tensor<elt_t> slow_range1(const Tensor<elt_t> &P, index i0, index i2,
                          index i1) {
  Indices i(1);
  i.at(0) = (i2 - i0) / i1 + 1;
  Tensor<elt_t> t(i);
  for (index i = i0, x = 0; i <= i2; i += i1, x++) {
    t.at(x) = P(i);
  }
  return t;
}

template <typename elt_t>
Tensor<elt_t> slow_range2(const Tensor<elt_t> &P, index i0, index i2, index i1,
                          index j0, index j2, index j1) {
  Indices i(2);
  i.at(0) = (i2 - i0) / i1 + 1;
  i.at(1) = (j2 - j0) / j1 + 1;
  Tensor<elt_t> t(i);
  for (index i = i0, x = 0; i <= i2; i += i1, x++) {
    for (index j = j0, y = 0; j <= j2; j += j1, y++) {
      t.at(x, y) = P(i, j);
    }
  }
  return t;
}

template <typename elt_t>
Tensor<elt_t> slow_range3(const Tensor<elt_t> &P, index i0, index i2, index i1,
                          index j0, index j2, index j1, index k0, index k2,
                          index k1) {
  Indices i(3);
  i.at(0) = (i2 - i0) / i1 + 1;
  i.at(1) = (j2 - j0) / j1 + 1;
  i.at(2) = (k2 - k0) / k1 + 1;
  Tensor<elt_t> t(i);
  for (index i = i0, x = 0; i <= i2; i += i1, x++) {
    for (index j = j0, y = 0; j <= j2; j += j1, y++) {
      for (index k = k0, z = 0; k <= k2; k += k1, z++) {
        t.at(x, y, z) = P(i, j, k);
      }
    }
  }
  return t;
}

template <typename elt_t>
Tensor<elt_t> slow_range4(const Tensor<elt_t> &P, index i0, index i2, index i1,
                          index j0, index j2, index j1, index k0, index k2,
                          index k1, index l0, index l2, index l1) {
  Indices i(4);
  i.at(0) = (i2 - i0) / i1 + 1;
  i.at(1) = (j2 - j0) / j1 + 1;
  i.at(2) = (k2 - k0) / k1 + 1;
  i.at(3) = (l2 - l0) / l1 + 1;
  Tensor<elt_t> t(i);
  for (index i = i0, x = 0; i <= i2; i += i1, x++) {
    for (index j = j0, y = 0; j <= j2; j += j1, y++) {
      for (index k = k0, z = 0; k <= k2; k += k1, z++) {
        for (index l = l0, w = 0; l <= l2; l += l1, z++) {
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
void test_extract_unit_size1(Tensor<elt_t> &P, index i) {
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
void test_extract_unit_size2(Tensor<elt_t> &P, index i, index j) {
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
void test_extract_unit_size3(Tensor<elt_t> &P, index i, index j, index k) {
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
void test_view_extract1(Tensor<elt_t> &P, index i0, index i2, index i1) {
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
void test_view_extract(Tensor<elt_t> &P, index i0, index i2, index i1, index j0,
                       index j2, index j1) {
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
void test_view_extract(Tensor<elt_t> &P, index i0, index i2, index i1, index j0,
                       index j2, index j1, index k0, index k2, index k1) {
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

  index d0 = P.dimension(0);
  for (index i = 0; i < d0; i++) {
    test_extract_unit_size1(P, i);
  }
  for (index i1 = 1; i1 < 4; i1++) {
    for (index i0 = 0; i0 < d0; i0++) {
      for (index i2 = i0; i2 < d0; i2++) {
        test_view_extract1(P, i0, i2, i1);
      }
    }
  }
}

template <typename elt_t>
void test_range_extract2(Tensor<elt_t> &P) {
  test_full_size_range2(P);

  index rows = P.dimension(0);
  index cols = P.dimension(1);
  for (index i = 0; i < rows; i++) {
    for (index j = 0; j < cols; j++) {
      test_extract_unit_size2(P, i, j);
    }
  }
  for (index i1 = 1; i1 < 4; i1++) {
    for (index j1 = 1; j1 < 4; j1++) {
      for (index i0 = 0; i0 < rows; i0++) {
        for (index j0 = 0; j0 < cols; j0++) {
          for (index i2 = i0; i2 < rows; i2++) {
            for (index j2 = j0; j2 < cols; j2++) {
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

  index d0 = P.dimension(0);
  index d1 = P.dimension(1);
  index d2 = P.dimension(2);
  for (index i = 0; i < d0; i++) {
    for (index j = 0; j < d1; j++) {
      for (index k = 0; k < d2; k++) {
        test_extract_unit_size3(P, i, j, k);
      }
    }
  }
  for (index i1 = 1; i1 < 3; i1++) {
    for (index j1 = 1; j1 < 3; j1++) {
      for (index k1 = 1; k1 < 3; k1++) {
        for (index i0 = 0; i0 < d0; i0++) {
          for (index j0 = 0; j0 < d1; j0++) {
            for (index k0 = 0; k0 < d2; k0++) {
              for (index i2 = i0; i2 < d0; i2++) {
                for (index j2 = j0; j2 < d1; j2++) {
                  for (index k2 = k0; k2 < d2; k2++) {
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
