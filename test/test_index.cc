// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include "alloc_informer.h"
#include <tensor/tensor.h>
#include <gtest/gtest.h>

using namespace tensor;

//////////////////////////////////////////////////////////////////////
// REFPOINTER
//

TEST(StaticIVectorTest, Size1) {
  Vector<tensor::index> v = igen << 3;
  EXPECT_EQ(1, v.size());
  EXPECT_EQ(3, v[0]);
  EXPECT_EQ(1, v.ref_count());
}

TEST(StaticIVectorTest, Size2) {
  Vector<tensor::index> v = igen << 3 << 4;
  EXPECT_EQ(2, v.size());
  EXPECT_EQ(3, v[0]);
  EXPECT_EQ(4, v[1]);
  EXPECT_EQ(1, v.ref_count());
}

TEST(StaticIndicesTest, Size1) {
  Indices v = igen << 3;
  EXPECT_EQ(1, v.size());
  EXPECT_EQ(3, v[0]);
  EXPECT_EQ(1, v.ref_count());
  Indices w = v;
  EXPECT_EQ(1, w.size());
}

TEST(StaticIndicesTest, Size2) {
  Indices v = igen << 3 << 4;
  EXPECT_EQ(2, v.size());
  EXPECT_EQ(3, v[0]);
  EXPECT_EQ(4, v[1]);
  EXPECT_EQ(1, v.ref_count());
  Indices w = v;
  EXPECT_EQ(2, w.size());
}

//
// REAL TENSORS
//

TEST(StaticRVectorTest, Size1) {
  Vector<double> v = rgen << 3.0;
  EXPECT_EQ(1, v.size());
  EXPECT_EQ(3, v[0]);
  EXPECT_EQ(1, v.ref_count());
}

TEST(StaticRVectorTest, Size2) {
  Vector<double> v = rgen << 3.0 << 4;
  EXPECT_EQ(2, v.size());
  EXPECT_EQ(3, v[0]);
  EXPECT_EQ(4, v[1]);
  EXPECT_EQ(1, v.ref_count());
}

TEST(StaticRTensorTest, Size1) {
  RTensor v = rgen << 3.0;
  EXPECT_EQ(1, v.size());
  EXPECT_EQ(3, v[0]);
  EXPECT_EQ(1, v.ref_count());
}

TEST(StaticRTensorTest, Size2) {
  RTensor v = rgen << 3.0 << 4;
  EXPECT_EQ(2, v.size());
  EXPECT_EQ(3, v[0]);
  EXPECT_EQ(4, v[1]);
  EXPECT_EQ(1, v.ref_count());
}

TEST(StaticRTensorTest, DataAndDim) {
  RTensor v(rgen << 1.0 << 2.0 << 3.0 << 4.0 << 5.0 << 6.0,
            igen << 2 << 3);
  EXPECT_EQ(6, v.size());
  EXPECT_EQ(2, v.rank());
  EXPECT_EQ(2, v.dimension(0));
  EXPECT_EQ(3, v.dimension(1));
  EXPECT_EQ(1.0, v[0]);
  EXPECT_EQ(2.0, v[1]);
  EXPECT_EQ(3.0, v[2]);
  EXPECT_EQ(4.0, v[3]);
  EXPECT_EQ(1.0, v(0,0));
  EXPECT_EQ(2.0, v(1,0));
  EXPECT_EQ(3.0, v(0,1));
  EXPECT_EQ(4.0, v(1,1));
  EXPECT_EQ(1, v.ref_count());
}

//
// COMPLEX TENSORS
//

TEST(StaticCTensorTest, Size1) {
  CTensor v = cgen << to_complex(3.0);
  EXPECT_EQ(1, v.size());
  EXPECT_EQ(to_complex(3), v[0]);
  EXPECT_EQ(1, v.ref_count());
}

TEST(StaticCTensorTest, Size2) {
  CTensor v = cgen << 3.0 << 4;
  EXPECT_EQ(2, v.size());
  EXPECT_EQ(to_complex(3.0), v[0]);
  EXPECT_EQ(to_complex(4.0), v[1]);
  EXPECT_EQ(1, v.ref_count());
}

TEST(StaticCTensorTest, DataAndDim) {
  CTensor v(cgen << 1.0 << 2.0 << 3.0 << 4.0 << 5.0 << 6.0,
            igen << 2 << 3);
  EXPECT_EQ(6, v.size());
  EXPECT_EQ(2, v.rank());
  EXPECT_EQ(2, v.dimension(0));
  EXPECT_EQ(3, v.dimension(1));
  EXPECT_EQ(1.0, v[0]);
  EXPECT_EQ(2.0, v[1]);
  EXPECT_EQ(3.0, v[2]);
  EXPECT_EQ(4.0, v[3]);
  EXPECT_EQ(1.0, v(0,0));
  EXPECT_EQ(2.0, v(1,0));
  EXPECT_EQ(3.0, v(0,1));
  EXPECT_EQ(4.0, v(1,1));
  EXPECT_EQ(1, v.ref_count());
}
