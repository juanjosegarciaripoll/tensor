// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include "loops.h"
#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>
#include <tensor/tensor.h>

namespace tensor_test {

  using namespace tensor;
  using tensor::index;

#include "test_view_common.cc"

  template<typename elt_t>
  void test_view_error(const Tensor &P)
  {
    if (P.rank() != 1) {
      ASSERT_DEATH(P(range()), ".*");
    }
    if (P.rank() != 2) {
      ASSERT_DEATH(P(range(), range()). ".*");
    }
    if (P.rank() != 3) {
      ASSERT_DEATH(P(range(), range(), range()). ".*");
    }
    if (P.rank() != 4) {
      ASSERT_DEATH(P(range(), range(), range(), range()). ".*");
    }
    if (P.rank() != 5) {
      ASSERT_DEATH(P(range(), range(), range(), range(), range()). ".*");
    }
    if (P.rank() != 6) {
      ASSERT_DEATH(P(range(), range(), range(), range(), range(), range()). ".*");
    }
  }

  //
  // REAL SPECIALIZATIONS
  //

  TEST(SliceTestError, SliceRTensorError0D) {
    test_view_error(RTensor());
  }

  TEST(SliceTestError, SliceRTensorError1D) {
    test_view_error(RTensor(gen << 3));
  }

  TEST(SliceTestError, SliceRTensorError2D) {
    test_view_error(RTensor(gen << 3 << 5));
  }

  TEST(SliceTestError, SliceRTensorError3D) {
    test_view_error(RTensor(gen << 2 << 4 << 5));
  }

  TEST(SliceTestError, SliceRTensorError4D) {
    test_view_error(RTensor(gen << 2 << 3 << 1 << 5));
  }

  TEST(SliceTestError, SliceRTensorError5D) {
    test_view_error(RTensor(gen << 2 << 1 << 4 << 3 << 5));
  }

  TEST(SliceTestError, SliceRTensorError6D) {
    test_view_error(RTensor(gen << 5 << 3 << 1 << 4 << 2 << 5));
  }

  //
  // COMPLEX SPECIALIZATIONS
  //

  TEST(SliceTestError, SliceCTensorError0D) {
    test_view_error(CTensor());
  }

  TEST(SliceTestError, SliceCTensorError1D) {
    test_view_error(CTensor(gen << 3));
  }

  TEST(SliceTestError, SliceCTensorError2D) {
    test_view_error(CTensor(gen << 3 << 5));
  }

  TEST(SliceTestError, SliceCTensorError3D) {
    test_view_error(CTensor(gen << 2 << 4 << 5));
  }

  TEST(SliceTestError, SliceCTensorError4D) {
    test_view_error(CTensor(gen << 2 << 3 << 1 << 5));
  }

  TEST(SliceTestError, SliceCTensorError5D) {
    test_view_error(CTensor(gen << 2 << 1 << 4 << 3 << 5));
  }

  TEST(SliceTestError, SliceCTensorError6D) {
    test_view_error(CTensor(gen << 5 << 3 << 1 << 4 << 2 << 5));
  }

} // namespace tensor_test
