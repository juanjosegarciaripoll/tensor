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

template <typename elt_t>
void test_view_error(const Tensor<elt_t> &P) {
  EXPECT_EQ(P(_).size(), P.size());
  if (P.rank() != 2) {
    ASSERT_THROW_DEBUG(P(_, _), std::out_of_range);
  }
  if (P.rank() != 3) {
    ASSERT_THROW_DEBUG(P(_, _, _), std::out_of_range);
  }
  if (P.rank() != 4) {
    ASSERT_THROW_DEBUG(P(_, _, _, _), std::out_of_range);
  }
  if (P.rank() != 5) {
    ASSERT_THROW_DEBUG(P(_, _, _, _, _), std::out_of_range);
  }
  if (P.rank() != 6) {
    ASSERT_THROW_DEBUG(P(_, _, _, _, _, _), std::out_of_range);
  }
}

//
// REAL SPECIALIZATIONS
//

#ifdef TENSOR_DEBUG
// death by assert

TEST(SliceTestError, SliceRTensorError0D) { test_view_error(RTensor()); }

TEST(SliceTestError, SliceRTensorError1D) {
  test_view_error(RTensor::empty(3));
}

TEST(SliceTestError, SliceRTensorError2D) {
  test_view_error(RTensor::empty(3, 5));
}

TEST(SliceTestError, SliceRTensorError3D) {
  test_view_error(RTensor::empty(2, 4, 5));
}

TEST(SliceTestError, SliceRTensorError4D) {
  test_view_error(RTensor::empty(2, 3, 1, 5));
}

TEST(SliceTestError, SliceRTensorError5D) {
  test_view_error(RTensor::empty(2, 4, 1, 3, 5));
}

TEST(SliceTestError, SliceRTensorError6D) {
  test_view_error(RTensor::empty(2, 1, 3, 5, 2, 4));
}

//
// COMPLEX SPECIALIZATIONS
//

TEST(SliceTestError, SliceCTensorError0D) { test_view_error(CTensor()); }

TEST(SliceTestError, SliceCTensorError1D) {
  test_view_error(CTensor::empty(3));
}

TEST(SliceTestError, SliceCTensorError2D) {
  test_view_error(CTensor::empty(3, 5));
}

TEST(SliceTestError, SliceCTensorError3D) {
  test_view_error(CTensor::empty(2, 4, 5));
}

TEST(SliceTestError, SliceCTensorError4D) {
  test_view_error(CTensor::empty(2, 3, 1, 5));
}

TEST(SliceTestError, SliceCTensorError5D) {
  test_view_error(CTensor::empty(2, 4, 1, 3, 5));
}

TEST(SliceTestError, SliceCTensorError6D) {
  test_view_error(CTensor::empty(2, 1, 3, 5, 2, 4));
}

#endif

}  // namespace tensor_test
