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

  //////////////////////////////////////////////////////////////////////
  // MANUALLY SET RANGES FROM A TENSOR USING LOOPS
  //

  template<typename elt_t> Tensor<elt_t>
  fill_continuous(const Tensor<elt_t> &P)
  {
    int m = 0;
    Tensor<elt_t> output(P.dimensions());
    for (typename Tensor<elt_t>::iterator it = output.begin(); it != output.end(); it++) {
      *it = number_zero<elt_t>() + (double)m++;
    }
    return output;
  }

  template<typename elt_t> Tensor<elt_t>
  slow_range_set1(Tensor<elt_t> &P,
                  index i0, index i2, index i1)
  {
    Indices i(1);
    i.at(0) = (i2 - i0) / i1 + 1;
    int n = 0;
    for (index i = i0, x = 0; i <= i2; i += i1, x++, n++) {
      P.at(x) = number_zero<elt_t>() + (double)n;
    }
  }

  template<typename elt_t> Tensor<elt_t>
  slow_range_set2(const Tensor<elt_t> &P,
                  index i0, index i2, index i1,
                  index j0, index j2, index j1)
  {
    Indices i(3);
    i.at(0) = (i2 - i0) / i1 + 1;
    i.at(1) = (j2 - j0) / j1 + 1;
    int n = 0;
    for (index i = i0, x = 0; i <= i2; i += i1, x++) {
      for (index j = j0, y = 0; j <= j2; j += j1, y++) {
        P.at(x,y) = number_zero<elt_t>() + (double)n++;
      }
    }
  }

  template<typename elt_t> void
  slow_range_set3(Tensor<elt_t> &P,
                  index i0, index i2, index i1,
                  index j0, index j2, index j1,
                  index k0, index k2, index k1)
  {
    Indices i(3);
    i.at(0) = (i2 - i0) / i1 + 1;
    i.at(1) = (j2 - j0) / j1 + 1;
    i.at(2) = (k2 - k0) / k1 + 1;
    int n = 0;
    for (index i = i0, x = 0; i <= i2; i += i1, x++) {
      for (index j = j0, y = 0; j <= j2; j += j1, y++) {
        for (index k = k0, z = 0; k <= k2; k += k1, z++) {
          P.at(x,y,z) = number_zero<elt_t>() + (double)n++;
        }
      }
    }
  }

  //////////////////////////////////////////////////////////////////////
  // FULL RANGE ASSIGNMENT
  //

  template<typename elt_t>
  void test_full_size_set1(Tensor<elt_t> &P) {
    Tensor<elt_t> Pcopy = P + number_zero<elt_t>();
    Tensor<elt_t> t = fill_continuous(P);
    P.at(range()) = t;
    EXPECT_EQ(P, t);
  }

  template<typename elt_t>
  void test_full_size_set2(Tensor<elt_t> &P) {
    Tensor<elt_t> Pcopy = P + number_zero<elt_t>();
    Tensor<elt_t> t = fill_continuous(P);
    P.at(range(),range()) = t;
    EXPECT_EQ(P, t);
  }

  template<typename elt_t>
  void test_full_size_set3(Tensor<elt_t> &P) {
    Tensor<elt_t> Pcopy = P + number_zero<elt_t>();
    Tensor<elt_t> t = fill_continuous(P);
    P.at(range(),range(),range()) = t;
    EXPECT_EQ(P, t);
  }

  /////////////////////////////////////////////////////////////////////
  // REAL SPECIALIZATIONS
  //

  TEST(SliceTest, SliceRTensor1DSetFull) {
    test_over_fixed_rank_tensors<double>(test_full_size_set1<double>,1);
  }

  TEST(SliceTest, SliceRTensor2DSetFull) {
    test_over_fixed_rank_tensors<double>(test_full_size_set2<double>,3);
  }

  TEST(SliceTest, SliceRTensor3DSetFull) {
    test_over_fixed_rank_tensors<double>(test_full_size_set3<double>,3);
  }

  /////////////////////////////////////////////////////////////////////
  // COMPLEX SPECIALIZATIONS
  //

  TEST(SliceTest, SliceCTensor1DSetFull) {
    test_over_fixed_rank_tensors<cdouble>(test_full_size_set1<cdouble>,1);
  }

  TEST(SliceTest, SliceCTensor2DSetFull) {
    test_over_fixed_rank_tensors<cdouble>(test_full_size_set2<cdouble>,3);
  }

  TEST(SliceTest, SliceCTensor3DSetFull) {
    test_over_fixed_rank_tensors<cdouble>(test_full_size_set3<cdouble>,3);
  }


} // namespace tensor_test
