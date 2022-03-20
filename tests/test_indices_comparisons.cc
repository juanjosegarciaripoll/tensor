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

namespace tensor_test {

//////////////////////////////////////////////////////////////////////
// INDICES SPECIALIZATIONS
//

TEST(IndicesCompare, IndicesIndicesEqual) {
  EXPECT_EQ(Booleans(), Indices() == Indices());

  EXPECT_EQ((Booleans{true}), (Indices{1} == Indices{1}));
  EXPECT_EQ((Booleans{false}), (Indices{1} == Indices{0}));

  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2} == Indices{1, 2}));
  EXPECT_EQ((Booleans{true, false}), (Indices{1, 2} == Indices{1, 1}));
  EXPECT_EQ((Booleans{false, true}), (Indices{1, 2} == Indices{2, 2}));
  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2} == Indices{2, 1}));
}

TEST(IndicesCompare, IndicesIndicesNotEqual) {
  EXPECT_EQ(Booleans(), Indices() != Indices());

  EXPECT_EQ((Booleans{false}), (Indices{1}) != (Indices{1}));
  EXPECT_EQ((Booleans{true}), (Indices{1}) != (Indices{0}));

  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2}) != (Indices{1, 2}));
  EXPECT_EQ((Booleans{false, true}), (Indices{1, 2}) != (Indices{1, 1}));
  EXPECT_EQ((Booleans{true, false}), (Indices{1, 2}) != (Indices{2, 2}));
  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2}) != (Indices{2, 1}));
}

TEST(IndicesCompare, IndicesIndicesLess) {
  EXPECT_EQ(Booleans(), Indices() < Indices());

  EXPECT_EQ((Booleans{false}), (Indices{1}) < (Indices{1}));
  EXPECT_EQ((Booleans{false}), (Indices{1}) < (Indices{0}));
  EXPECT_EQ((Booleans{true}), (Indices{1}) < (Indices{2}));

  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2}) < (Indices{1, 2}));
  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2}) < (Indices{1, 1}));
  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2}) < (Indices{0, 2}));
  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2}) < (Indices{0, 0}));
  EXPECT_EQ((Booleans{true, false}), (Indices{1, 2}) < (Indices{2, 1}));
  EXPECT_EQ((Booleans{false, true}), (Indices{1, 2}) < (Indices{1, 3}));
  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2}) < (Indices{2, 3}));
}

TEST(IndicesCompare, IndicesIndicesGreater) {
  EXPECT_EQ(Booleans(), Indices() > Indices());

  EXPECT_EQ((Booleans{false}), (Indices{1}) > (Indices{1}));
  EXPECT_EQ((Booleans{true}), (Indices{1}) > (Indices{0}));
  EXPECT_EQ((Booleans{false}), (Indices{1}) > (Indices{2}));

  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2}) > (Indices{1, 2}));
  EXPECT_EQ((Booleans{false, true}), (Indices{1, 2}) > (Indices{1, 1}));
  EXPECT_EQ((Booleans{true, false}), (Indices{1, 2}) > (Indices{0, 2}));
  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2}) > (Indices{0, 0}));
  EXPECT_EQ((Booleans{false, true}), (Indices{1, 2}) > (Indices{2, 1}));
  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2}) > (Indices{1, 3}));
  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2}) > (Indices{2, 3}));
}

TEST(IndicesCompare, IndicesIndicesLessEqual) {
  EXPECT_EQ(Booleans(), Indices() <= Indices());

  EXPECT_EQ((Booleans{true}), (Indices{1}) <= (Indices{1}));
  EXPECT_EQ((Booleans{false}), (Indices{1}) <= (Indices{0}));
  EXPECT_EQ((Booleans{true}), (Indices{1}) <= (Indices{2}));

  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2}) <= (Indices{1, 2}));
  EXPECT_EQ((Booleans{true, false}), (Indices{1, 2}) <= (Indices{1, 1}));
  EXPECT_EQ((Booleans{false, true}), (Indices{1, 2}) <= (Indices{0, 2}));
  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2}) <= (Indices{0, 0}));
  EXPECT_EQ((Booleans{true, false}), (Indices{1, 2}) <= (Indices{2, 1}));
  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2}) <= (Indices{1, 3}));
  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2}) <= (Indices{2, 3}));
}

TEST(IndicesCompare, IndicesIndicesGreaterEqual) {
  EXPECT_EQ(Booleans(), Indices() >= Indices());

  EXPECT_EQ((Booleans{true}), (Indices{1}) >= (Indices{1}));
  EXPECT_EQ((Booleans{true}), (Indices{1}) >= (Indices{0}));
  EXPECT_EQ((Booleans{false}), (Indices{1}) >= (Indices{2}));

  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2}) >= (Indices{1, 2}));
  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2}) >= (Indices{1, 1}));
  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2}) >= (Indices{0, 2}));
  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2}) >= (Indices{0, 0}));
  EXPECT_EQ((Booleans{false, true}), (Indices{1, 2}) >= (Indices{2, 1}));
  EXPECT_EQ((Booleans{true, false}), (Indices{1, 2}) >= (Indices{1, 3}));
  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2}) >= (Indices{2, 3}));
}

//////////////////////////////////////////////////////////////////////
// INDICES - INTEGER SPECIALIZATIONS
//

TEST(IndicesCompare, IndicesIndexEqual) {
  EXPECT_EQ(Booleans(), Indices() == 0);

  EXPECT_EQ((Booleans{true}), (Indices{1}) == 1);
  EXPECT_EQ((Booleans{false}), (Indices{1}) == 0);

  EXPECT_EQ((Booleans{true, false}), (Indices{1, 2}) == 1);
  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2}) == 0);
  EXPECT_EQ((Booleans{false, true}), (Indices{1, 2}) == 2);
}

TEST(IndicesCompare, IndicesIndexNotEqual) {
  EXPECT_EQ(Booleans(), Indices() != 0);

  EXPECT_EQ((Booleans{false}), (Indices{1}) != 1);
  EXPECT_EQ((Booleans{true}), (Indices{1}) != 0);

  EXPECT_EQ((Booleans{false, true}), (Indices{1, 2}) != 1);
  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2}) != 0);
  EXPECT_EQ((Booleans{true, false}), (Indices{1, 2}) != 2);
}

TEST(IndicesCompare, IndicesIndexLess) {
  EXPECT_EQ(Booleans(), Indices() < 1);

  EXPECT_EQ((Booleans{false}), (Indices{1}) < 0);
  EXPECT_EQ((Booleans{false}), (Indices{1}) < 1);
  EXPECT_EQ((Booleans{true}), (Indices{1}) < 2);

  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2}) < 0);
  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2}) < 1);
  EXPECT_EQ((Booleans{true, false}), (Indices{1, 2}) < 2);
  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2}) < 3);
}

TEST(IndicesCompare, IndicesIndexGreater) {
  EXPECT_EQ(Booleans(), Indices() > 1);

  EXPECT_EQ((Booleans{true}), (Indices{1}) > 0);
  EXPECT_EQ((Booleans{false}), (Indices{1}) > 1);
  EXPECT_EQ((Booleans{false}), (Indices{1}) > 2);

  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2}) > 0);
  EXPECT_EQ((Booleans{false, true}), (Indices{1, 2}) > 1);
  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2}) > 2);
  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2}) > 3);
}

TEST(IndicesCompare, IndicesIndexLessEqual) {
  EXPECT_EQ(Booleans(), Indices() <= 1);

  EXPECT_EQ((Booleans{false}), (Indices{1}) <= 0);
  EXPECT_EQ((Booleans{true}), (Indices{1}) <= 1);
  EXPECT_EQ((Booleans{true}), (Indices{1}) <= 2);

  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2}) <= 0);
  EXPECT_EQ((Booleans{true, false}), (Indices{1, 2}) <= 1);
  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2}) <= 2);
  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2}) <= 3);
}

TEST(IndicesCompare, IndicesIndexGreaterEqual) {
  EXPECT_EQ(Booleans(), Indices() >= 1);

  EXPECT_EQ((Booleans{true}), (Indices{1}) >= 0);
  EXPECT_EQ((Booleans{true}), (Indices{1}) >= 1);
  EXPECT_EQ((Booleans{false}), (Indices{1}) >= 2);

  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2}) >= 0);
  EXPECT_EQ((Booleans{true, true}), (Indices{1, 2}) >= 1);
  EXPECT_EQ((Booleans{false, true}), (Indices{1, 2}) >= 2);
  EXPECT_EQ((Booleans{false, false}), (Indices{1, 2}) >= 3);
}

}  // namespace tensor_test
