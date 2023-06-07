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

#include <gtest/gtest.h>
#include <tensor/indices.h>


namespace tensor_test {

using namespace tensor;

TEST(Dimensions, equality) {
  Dimensions base{1, 2, 3};

  Dimensions same{1, 2, 3};
  Dimensions otherSize{1, 2, 4};
  Dimensions otherRank{1, 2, 3, 1};

  ASSERT_TRUE(base == same);
  ASSERT_FALSE(base == otherSize);
  ASSERT_FALSE(base == otherRank);

  ASSERT_FALSE(base != same);
  ASSERT_TRUE(base != otherSize);
  ASSERT_TRUE(base != otherRank);
}

}
