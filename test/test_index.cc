// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include "alloc_informer.h"
#include <tensor/tensor.h>
#include <gtest/gtest.h>

using namespace tensor;

tensor::ListGenerator tensor::gen;

//////////////////////////////////////////////////////////////////////
// REFPOINTER
//

TEST(StaticIVectorTest, Size1) {
  Vector<tensor::index> v = gen >> 3;
  EXPECT_EQ(1, v.size());
  EXPECT_EQ(3, v[0]);
  EXPECT_EQ(1, v.ref_count());
}

TEST(StaticIVectorTest, Size2) {
  Vector<tensor::index> v = gen >> 3 >> 4;
  EXPECT_EQ(2, v.size());
  EXPECT_EQ(3, v[0]);
  EXPECT_EQ(4, v[1]);
  EXPECT_EQ(1, v.ref_count());
}

/*
TEST(StaticIndicesTest, Size1) {
  Vector<tensor::index> v = gen >> 3;
  EXPECT_EQ(1, v.size());
  EXPECT_EQ(3, v[0]);
  EXPECT_EQ(1, v.ref_count());
  Indices w = v;
  EXPECT_EQ(1, w.size());
}

TEST(StaticIndicesTest, Size2) {
  Indices v = gen >> 3 >> 4;
  EXPECT_EQ(2, v.size());
  EXPECT_EQ(3, v[0]);
  EXPECT_EQ(4, v[1]);
  EXPECT_EQ(1, v.ref_count());
  Indices w = v;
  EXPECT_EQ(2, w.size());
}
*/

TEST(StaticRVectorTest, Size1) {
  Vector<double> v = gen >> 3.0;
  EXPECT_EQ(1, v.size());
  EXPECT_EQ(3, v[0]);
  EXPECT_EQ(1, v.ref_count());
}

TEST(StaticRVectorTest, Size2) {
  Vector<double> v = gen >> 3.0 >> 4;
  EXPECT_EQ(2, v.size());
  EXPECT_EQ(3, v[0]);
  EXPECT_EQ(4, v[1]);
  EXPECT_EQ(1, v.ref_count());
}

/*

TEST(StaticRTensorTest, Size1) {
  RTensor v = gen >> 3.0;
  EXPECT_EQ(1, v.size());
  EXPECT_EQ(3, v[0]);
  EXPECT_EQ(1, v.ref_count());
}

TEST(StaticRTensorTest, Size2) {
  RTensor v = gen >> 3.0 >> 4;
  EXPECT_EQ(2, v.size());
  EXPECT_EQ(3, v[0]);
  EXPECT_EQ(4, v[1]);
  EXPECT_EQ(1, v.ref_count());
}

*/
