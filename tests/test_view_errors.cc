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

using namespace tensor;
using tensor::index;

#include "test_view_common.cc"

template <typename elt_t>
void test_view_error(const Tensor<elt_t> &P) {
  if (P.rank() < 1) {
    ASSERT_THROW(P(range()), std::out_of_range);
  } else {
    EXPECT_EQ(P(range()).size(), P.size());
  }
  if (P.rank() != 2) {
    ASSERT_THROW(P(range(), range()), std::out_of_range);
  }
  if (P.rank() != 3) {
    ASSERT_THROW(P(range(), range(), range()), std::out_of_range);
  }
  if (P.rank() != 4) {
    ASSERT_THROW(P(range(), range(), range(), range()), std::out_of_range);
  }
  if (P.rank() != 5) {
    ASSERT_THROW(P(range(), range(), range(), range(), range()),
                 std::out_of_range);
  }
  if (P.rank() != 6) {
    ASSERT_THROW(P(range(), range(), range(), range(), range(), range()),
                 std::out_of_range);
  }
}

//
// REAL SPECIALIZATIONS
//

#ifndef NDEBUG
// death by assert

TEST(SliceTestError, SliceRTensorError0D) { test_view_error(RTensor()); }

TEST(SliceTestError, SliceRTensorError1D) {
  test_view_error(RTensor(rgen << 3));
}

TEST(SliceTestError, SliceRTensorError2D) {
  test_view_error(RTensor(rgen << 3 << 5));
}

TEST(SliceTestError, SliceRTensorError3D) {
  test_view_error(RTensor(rgen << 2 << 4 << 5));
}

TEST(SliceTestError, SliceRTensorError4D) {
  test_view_error(RTensor(rgen << 2 << 3 << 1 << 5));
}

TEST(SliceTestError, SliceRTensorError5D) {
  test_view_error(RTensor(rgen << 2 << 1 << 4 << 3 << 5));
}

TEST(SliceTestError, SliceRTensorError6D) {
  test_view_error(RTensor(rgen << 5 << 3 << 1 << 4 << 2 << 5));
}

//
// COMPLEX SPECIALIZATIONS
//

TEST(SliceTestError, SliceCTensorError0D) { test_view_error(CTensor()); }

TEST(SliceTestError, SliceCTensorError1D) {
  test_view_error(CTensor(cgen << 3));
}

TEST(SliceTestError, SliceCTensorError2D) {
  test_view_error(CTensor(cgen << 3 << 5));
}

TEST(SliceTestError, SliceCTensorError3D) {
  test_view_error(CTensor(cgen << 2 << 4 << 5));
}

TEST(SliceTestError, SliceCTensorError4D) {
  test_view_error(CTensor(cgen << 2 << 3 << 1 << 5));
}

TEST(SliceTestError, SliceCTensorError5D) {
  test_view_error(CTensor(cgen << 2 << 1 << 4 << 3 << 5));
}

TEST(SliceTestError, SliceCTensorError6D) {
  test_view_error(CTensor(cgen << 5 << 3 << 1 << 4 << 2 << 5));
}

#endif

}  // namespace tensor_test
