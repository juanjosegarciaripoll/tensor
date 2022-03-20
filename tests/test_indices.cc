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
#include <tensor/indices.h>

namespace tensor_test {

using namespace tensor;
using tensor::index;

TEST(Indices, EmptyIndices) {
  Indices ndx;
  ASSERT_EQ(ndx.size(), 0);
}

TEST(Indices, BracedInitialization) {
  {
    Indices ndx = {1};
    ASSERT_EQ(ndx.size(), 1);
    ASSERT_EQ(ndx[0], 1);
  }
  {
    Indices ndx = {1, 3, 5};  // NOLINT
    ASSERT_EQ(ndx.size(), 3);
    ASSERT_EQ(ndx[0], 1);
    ASSERT_EQ(ndx[1], 3);
    ASSERT_EQ(ndx[2], 5);
  }
}

TEST(Indices, FromVector) {
  {
    Vector<index> v(3);
    v.at(0) = 1;
    v.at(1) = 2;
    v.at(2) = 3;
    Indices ndx = v;
    ASSERT_EQ(ndx.size(), 3);
    ASSERT_EQ(ndx[0], 1);
    ASSERT_EQ(ndx[1], 2);
    ASSERT_EQ(ndx[2], 3);
  }
}

TEST(Indices, MakeRange) {
  {
    Indices ndx = Indices::range(0, -3, 1);
    ASSERT_EQ(ndx.size(), 0);
  }
  {
    Indices ndx = Indices::range(0, 3, 1);
    ASSERT_EQ(ndx.size(), 4);
    ASSERT_EQ(ndx[0], 0);
    ASSERT_EQ(ndx[1], 1);
    ASSERT_EQ(ndx[2], 2);
    ASSERT_EQ(ndx[3], 3);
  }
  {
    Indices ndx = Indices::range(0, 3, 2);
    ASSERT_EQ(ndx.size(), 2);
    ASSERT_EQ(ndx[0], 0);
    ASSERT_EQ(ndx[1], 2);
  }
  {
    Indices ndx = Indices::range(0, 3, 3);
    ASSERT_EQ(ndx.size(), 2);
    ASSERT_EQ(ndx[0], 0);
    ASSERT_EQ(ndx[1], 3);
  }
  {
    Indices ndx = Indices::range(0, 3, 4);
    ASSERT_EQ(ndx.size(), 1);
    ASSERT_EQ(ndx[0], 0);
  }
}

}  // namespace tensor_test
