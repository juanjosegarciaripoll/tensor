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

#include "alloc_informer.h"
#include <tensor/refcount.h>
#include <gtest/gtest.h>

using tensor::RefPointer;

//////////////////////////////////////////////////////////////////////
// REFPOINTER
//

TEST(RefPointerTest, DefaultConstructor) {
  const RefPointer<int> r;
  EXPECT_EQ(0, r.size());
  EXPECT_FALSE(r.begin_const());
  EXPECT_FALSE(r.other_references());
}

// Verify proper size of object and that the exact number of elements
// are allocated.
TEST(RefPointerTest, SizeConstructor) {
  for (int i = 1; i < 10; ++i) {
    {
      AllocInformer::reset_counters();
      const RefPointer<AllocInformer> r(i);
      EXPECT_EQ(i, r.size());
      EXPECT_EQ(1, r.ref_count());
      EXPECT_EQ(i, AllocInformer::allocations);
    }
    EXPECT_EQ(i, AllocInformer::deallocations);
  }
}

// For a constant object with a single reference, the pointer is
// always the same and does not change.
TEST(RefPointerTest, SingleConstantReference) {
  const RefPointer<int> r(2);
  const int *p = r.begin_const();
  EXPECT_EQ(p, r.begin());
  EXPECT_EQ(1, r.ref_count());
}

// For a non constant object with a single reference, the pointer is
// always the same and does not change.
TEST(RefPointerTest, SingleReferenceAppropiate) {
  RefPointer<int> r(2);
  const int *p = r.begin_const();
  // Appropiate does not change the pointer
  r.appropiate();
  EXPECT_EQ(p, r.begin_const());
}

// For a non constant object with a single reference, the pointer is
// always the same and does not change, nor does getting a non
// constant reference
TEST(RefPointerTest, SingleReferencePointer) {
  RefPointer<int> r(2);
  const int *p = r.begin_const();
  EXPECT_EQ(p, r.begin());
  EXPECT_EQ(p, r.begin_const());
}

// The copy constructor increases the number of references, so that
// r1 and r2 point to the same data.
TEST(RefPointerTest, TwoRefsCopyConstructor) {
  RefPointer<int> r1(2);
  const int *p = r1.begin_const();
  RefPointer<int> r2(r1);
  EXPECT_EQ(2, r1.size());
  EXPECT_EQ(2, r1.ref_count());
  EXPECT_EQ(2, r2.size());
  EXPECT_EQ(r1.begin_const(), r2.begin_const());
}

// When another reference appropiates of data, it creates a fresh new
// copy and does not affect the original one.
TEST(RefPointerTest, TwoRefsAppropriate) {
  RefPointer<int> r1(3);
  const int *p = r1.begin_const();
  RefPointer<int> r2(r1);
  r2.appropiate();
  EXPECT_EQ(3, r1.size());
  EXPECT_EQ(p, r1.begin_const());
  EXPECT_EQ(1, r1.ref_count());
  EXPECT_EQ(3, r2.size());
  EXPECT_NE(p, r2.begin_const());
  EXPECT_EQ(1, r2.ref_count());
}
