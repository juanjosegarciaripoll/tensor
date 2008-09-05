// -*- mode: c++; fill-column: 80; c-basic-offset: 4; -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <gtest/gtest.h>
#include <tensor/refcount.h>

TEST(RefcountTest, DefaultConstructor) {
  const RefPointer<int> r;
  EXPECT_EQ(0, r.size());
  EXPECT_FALSE(r.constant_pointer());
  EXPECT_EQ(0, r.ref_count());
}

TEST(RefcountTest, SingleConstantReference) {
  // For a constant object with a single reference, the pointer is
  // always the same and does not change.
  const RefPointer<int> r(2);
  EXPECT_EQ(2, r.size());
  EXPECT_EQ(1, r.ref_count());
  const int *p = r.constant_pointer();
  EXPECT_EQ(p, r.pointer());
}

TEST(RefcountTest, SingleReference) {
  // For a non constant object with a single reference, the pointer is
  // always the same and does not change.
  RefPointer<int> r(2);
  EXPECT_EQ(2, r.size());
  EXPECT_EQ(1, r.ref_count());
  const int *p = r.constant_pointer();
  // Appropiate does not change the pointer
  r.appropiate();
  EXPECT_EQ(p, r.constant_pointer());
  // Nor does getting a non constant reference
  EXPECT_EQ(p, r.pointer());
}

TEST(RefcountTest, TwoRefsCopyConstructor) {
  RefPointer<int> r1(2);
  const int *p = r1.constant_pointer();
  // The copy constructor increases the number of references, so that
  // r1 and r2 point to the same data.
  RefPointer<int> r2(r1);
  EXPECT_EQ(2, r1.size());
  EXPECT_EQ(2, r1.ref_count());
  EXPECT_EQ(2, r2.size());
  EXPECT_EQ(r1.constant_pointer(), r2.constant_pointer());
}

TEST(RefcountTest, TwoRefsAppropriate) {
  // When r2 appropiates of data, it creates a fresh new copy.
  RefPointer<int> r1(2);
  const int *p = r1.constant_pointer();
  RefPointer<int> r2(r1);
  r2.appropiate();
  EXPECT_EQ(2, r1.size());
  EXPECT_EQ(p, r1.constant_pointer());
  EXPECT_EQ(2, r2.size());
  EXPECT_NE(p, r2.constant_pointer());
}
