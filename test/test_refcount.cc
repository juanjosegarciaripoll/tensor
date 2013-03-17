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
  EXPECT_EQ(0, r.begin_const());
  EXPECT_EQ(1, r.ref_count());
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

// When the pointer is initialized with some data, it points to this data.
// No additional data is created, but on destruction it should free the data.
TEST(RefPointerTest, DataConstructor) {
  const size_t size = 5;
  AllocInformer *data = new AllocInformer[size];
  AllocInformer::reset_counters();

  {
    const RefPointer<AllocInformer> r(data, size);
    EXPECT_EQ(0, AllocInformer::allocations);
    EXPECT_EQ(size, r.size());
    EXPECT_EQ(1, r.ref_count());
  }
  EXPECT_EQ(size, AllocInformer::deallocations);
}

// The copy constructor increases the number of references, so that
// r1 and r2 point to the same data.
TEST(RefPointerTest, TwoRefsCopyConstructor) {
  RefPointer<int> r1(2);
  RefPointer<int> r2(r1);
  EXPECT_EQ(2, r1.size());
  EXPECT_EQ(2, r1.ref_count());
  EXPECT_EQ(2, r2.size());
  EXPECT_EQ(r1.begin_const(), r2.begin_const());
}

// The destructor frees the data as soon as the reference count goes to zero.
TEST(RefPointerTest, Destructor) {
  AllocInformer::reset_counters();
  size_t size = 5;

  {
    RefPointer<AllocInformer> r1(size);
    {
      RefPointer<AllocInformer> r2(r1);
      EXPECT_EQ(2, r2.ref_count());
    }
    EXPECT_EQ(0, AllocInformer::deallocations);
  }
  EXPECT_EQ(size, AllocInformer::deallocations);
}

// Operator= makes two references point to the same data
TEST(RefPointerTest, assigning) {
  RefPointer<int> ref(5);
  RefPointer<int> r2 = ref;

  EXPECT_EQ(ref.begin_const(), r2.begin_const());
}

// For constant pointer access, no data is copied; multiple references
// view the same data.
TEST(RefPointerTest, ConstantAccess) {
  RefPointer<int> ref(2);
  const RefPointer<int> const_ref(ref);

  const int *start = ref.begin_const();
  const int *end = ref.end_const();

  EXPECT_EQ(start, const_ref.begin_const());
  EXPECT_EQ(start, const_ref.begin());
  EXPECT_EQ(end, const_ref.end_const());
  EXPECT_EQ(end, const_ref.end());
}

// As soon as a non-const pointer is requested, the data will probably
// be modified. Check that the data is then copied.
TEST(RefPointerTest, NonConstAccess) {
  RefPointer<int> ref(2);
  RefPointer<int> start_ref(ref);
  RefPointer<int> end_ref(ref);

  // intially, all pointers point to the same position
  EXPECT_EQ(ref.begin_const(), start_ref.begin_const());
  EXPECT_EQ(ref.end_const(), end_ref.end_const());
  EXPECT_EQ(3, ref.ref_count());

  // now this changes.
  EXPECT_NE(ref.begin_const(), start_ref.begin());
  EXPECT_EQ(2, ref.ref_count());
  EXPECT_NE(ref.end_const(), end_ref.end());
  EXPECT_EQ(1, ref.ref_count());

  // and all the data was of course copied.
  EXPECT_NE(ref.end_const(), start_ref.end_const());
  EXPECT_NE(ref.begin_const(), end_ref.begin_const());
}

// For a single object, no data is copied ever.
TEST(RefPointerTest, SingleReference) {
  RefPointer<int> r(2);
  const int *p = r.begin_const();

  EXPECT_EQ(p, r.begin());
  EXPECT_EQ(p, r.begin_const());
  EXPECT_EQ(1, r.ref_count());
}

// Reallocating completely sets a reference to new data.
TEST(RefPointerTest, Reallocation) {
  RefPointer<int> r(2);
  RefPointer<int> newPointer(r);
  int newsize = 5;

  EXPECT_EQ(2, r.ref_count());
  EXPECT_EQ(r.begin_const(), newPointer.begin_const());

  newPointer.reallocate(newsize);

  EXPECT_EQ(1, r.ref_count());
  EXPECT_EQ(1, newPointer.ref_count());
  EXPECT_NE(r.begin_const(), newPointer.begin_const());
  EXPECT_EQ(newsize, newPointer.size());
}