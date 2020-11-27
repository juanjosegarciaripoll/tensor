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
  RTensor v(rgen << 1.0 << 2.0 << 3.0 << 4.0 << 5.0 << 6.0, igen << 2 << 3);
  EXPECT_EQ(6, v.size());
  EXPECT_EQ(2, v.rank());
  EXPECT_EQ(2, v.dimension(0));
  EXPECT_EQ(3, v.dimension(1));
  EXPECT_EQ(1.0, v[0]);
  EXPECT_EQ(2.0, v[1]);
  EXPECT_EQ(3.0, v[2]);
  EXPECT_EQ(4.0, v[3]);
  EXPECT_EQ(1.0, v(0, 0));
  EXPECT_EQ(2.0, v(1, 0));
  EXPECT_EQ(3.0, v(0, 1));
  EXPECT_EQ(4.0, v(1, 1));
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
  CTensor v(cgen << 1.0 << 2.0 << 3.0 << 4.0 << 5.0 << 6.0, igen << 2 << 3);
  EXPECT_EQ(6, v.size());
  EXPECT_EQ(2, v.rank());
  EXPECT_EQ(2, v.dimension(0));
  EXPECT_EQ(3, v.dimension(1));
  EXPECT_EQ(1.0, v[0]);
  EXPECT_EQ(2.0, v[1]);
  EXPECT_EQ(3.0, v[2]);
  EXPECT_EQ(4.0, v[3]);
  EXPECT_EQ(1.0, v(0, 0));
  EXPECT_EQ(2.0, v(1, 0));
  EXPECT_EQ(3.0, v(0, 1));
  EXPECT_EQ(4.0, v(1, 1));
  EXPECT_EQ(1, v.ref_count());
}
