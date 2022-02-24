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
#include <tensor/vector.h>
#include <gtest/gtest.h>

using tensor::Vector;

//////////////////////////////////////////////////////////////////////
// REFPOINTER
//

TEST(VectorTest, DefaultConstructor) {
  const Vector<int> r;
  EXPECT_EQ(0, r.size());
  EXPECT_EQ(0, r.cbegin());
#ifdef TENSOR_REFCOUNT_H
  EXPECT_EQ(1, r.ref_count());
#endif
}

// Verify proper size of object and that the exact number of elements
// are allocated.
TEST(VectorTest, SizeConstructor) {
  for (tensor::index i = 1; i < 10; ++i) {
    {
      AllocInformer::reset_counters();
      const Vector<AllocInformer> r(i);
      EXPECT_EQ(i, r.size());
      EXPECT_EQ(1, r.ref_count());
      EXPECT_EQ(i, AllocInformer::allocations);
    }
    EXPECT_EQ(i, AllocInformer::deallocations);
  }
}
/*
// When the pointer is initialized with some data, it points to this data.
// No additional data is created, but on destruction it should free the data.
TEST(VectorTest, DataConstructor) {
  const size_t size = 5;
  AllocInformer *data = new AllocInformer[size];
  AllocInformer::reset_counters();

  {
    const Vector<AllocInformer> r(data, size);
    EXPECT_EQ(0, AllocInformer::allocations);
    EXPECT_EQ(size, r.size());
    EXPECT_EQ(1, r.ref_count());
  }
  EXPECT_EQ(size, AllocInformer::deallocations);
}
*/

// The copy constructor increases the number of references, so that
// r1 and r2 point to the same data.
TEST(VectorTest, TwoRefsCopyConstructor) {
  Vector<int> r1(size_t(2));
  Vector<int> r2(r1);
  EXPECT_EQ(2, r1.size());
  EXPECT_EQ(2, r1.ref_count());
  EXPECT_EQ(2, r2.size());
  EXPECT_EQ(r1.cbegin(), r2.cbegin());
}

// The destructor frees the data as soon as the reference count goes to zero.
TEST(VectorTest, Destructor) {
  AllocInformer::reset_counters();
  size_t size = 5;

  {
    Vector<AllocInformer> r1(size);
    {
      Vector<AllocInformer> r2(r1);
      EXPECT_EQ(2, r2.ref_count());
    }
    EXPECT_EQ(0, AllocInformer::deallocations);
  }
  EXPECT_EQ(size, AllocInformer::deallocations);
}

// Operator= makes two references point to the same data
TEST(VectorTest, Assigning) {
  Vector<int> ref(size_t(5));
  Vector<int> r2 = ref;

  EXPECT_EQ(ref.cbegin(), r2.cbegin());
}

// Operator= works when a refpointer is copied onto itself
TEST(VectorTest, AssigningSelf) {
  Vector<int> ref(1);
  ref = ref;
  EXPECT_EQ(ref.ref_count(), 1);
}

// For constant pointer access, no data is copied; multiple references
// view the same data.
TEST(VectorTest, ConstantAccess) {
  Vector<int> ref(2);
  const Vector<int> const_ref(ref);

  const int *start = ref.cbegin();
  const int *end = ref.cend();

  EXPECT_EQ(start, const_ref.cbegin());
  EXPECT_EQ(start, const_ref.begin());
  EXPECT_EQ(end, const_ref.cend());
  EXPECT_EQ(end, const_ref.end());
}

// As soon as a non-const pointer is requested, the data will probably
// be modified. Check that the data is then copied.
TEST(VectorTest, NonConstAccess) {
  Vector<int> ref(2);
  Vector<int> start_ref(ref);
  Vector<int> end_ref(ref);

  // intially, all pointers point to the same position
  EXPECT_EQ(ref.cbegin(), start_ref.cbegin());
  EXPECT_EQ(ref.cend(), end_ref.cend());
  EXPECT_EQ(3, ref.ref_count());

  // now this changes.
  EXPECT_NE(ref.cbegin(), start_ref.begin());
  EXPECT_EQ(2, ref.ref_count());
  EXPECT_NE(ref.cend(), end_ref.end());
  EXPECT_EQ(1, ref.ref_count());

  // and all the data was of course copied.
  EXPECT_NE(ref.cend(), start_ref.cend());
  EXPECT_NE(ref.cbegin(), end_ref.cbegin());
}

// For a single object, no data is copied ever.
TEST(VectorTest, SingleReference) {
  Vector<int> r(2);
  const int *p = r.cbegin();

  EXPECT_EQ(p, r.begin());
  EXPECT_EQ(p, r.cbegin());
  EXPECT_EQ(1, r.ref_count());
}
