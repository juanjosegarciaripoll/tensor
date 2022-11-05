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
#include <tensor/tensor/accumulate.h>

namespace tensor_test {

TEST(RTensorTest, Sum1D) {
  EXPECT_CEQ(sum(RTensor{1.0, 2.0}, 0), RTensor{3.0});
  EXPECT_CEQ(sum(RTensor{1.0, 2.0}, -1), RTensor{3.0});

  EXPECT_CEQ(sum(RTensor{{1.0, 2.0}, {3.0, 4.0}}, 0), RTensor({4.0, 6.0}));
  EXPECT_CEQ(sum(RTensor{{1.0, 2.0}, {3.0, 4.0}}, 1), RTensor({3.0, 7.0}));
  EXPECT_CEQ(sum(RTensor{{1.0, 2.0}, {3.0, 4.0}}, -2), RTensor({4.0, 6.0}));
  EXPECT_CEQ(sum(RTensor{{1.0, 2.0}, {3.0, 4.0}}, -1), RTensor({3.0, 7.0}));
}

TEST(RTensorTest, Mean1D) {
  EXPECT_CEQ(mean(RTensor{1.0, 2.0}, 0), RTensor{1.5});
  EXPECT_CEQ(mean(RTensor{1.0, 2.0}, -1), RTensor{1.5});

  EXPECT_CEQ(mean(RTensor{{1.0, 2.0}, {3.0, 4.0}}, 0), RTensor({2.0, 3.0}));
  EXPECT_CEQ(mean(RTensor{{1.0, 2.0}, {3.0, 4.0}}, 1), RTensor({1.5, 3.5}));
  EXPECT_CEQ(mean(RTensor{{1.0, 2.0}, {3.0, 4.0}}, -2), RTensor({2.0, 3.0}));
  EXPECT_CEQ(mean(RTensor{{1.0, 2.0}, {3.0, 4.0}}, -1), RTensor({1.5, 3.5}));
}

TEST(RTensorTest, Max1D) {
  EXPECT_CEQ(max(RTensor{1.0, 2.0}, 0), RTensor{2.0});
  EXPECT_CEQ(max(RTensor{1.0, 2.0}, -1), RTensor{2.0});

  EXPECT_CEQ(max(RTensor{{1.0, 2.0}, {3.0, 4.0}}, 0), RTensor({3.0, 4.0}));
  EXPECT_CEQ(max(RTensor{{1.0, 2.0}, {3.0, 4.0}}, 1), RTensor({2.0, 4.0}));
  EXPECT_CEQ(max(RTensor{{1.0, 2.0}, {3.0, 4.0}}, -2), RTensor({3.0, 4.0}));
  EXPECT_CEQ(max(RTensor{{1.0, 2.0}, {3.0, 4.0}}, -1), RTensor({2.0, 4.0}));
}

TEST(RTensorTest, Min1D) {
  EXPECT_CEQ(min(RTensor{1.0, 2.0}, 0), RTensor{1.0});
  EXPECT_CEQ(min(RTensor{1.0, 2.0}, -1), RTensor{1.0});

  EXPECT_CEQ(min(RTensor{{1.0, 2.0}, {3.0, 4.0}}, 0), RTensor({1.0, 2.0}));
  EXPECT_CEQ(min(RTensor{{1.0, 2.0}, {3.0, 4.0}}, 1), RTensor({1.0, 3.0}));
  EXPECT_CEQ(min(RTensor{{1.0, 2.0}, {3.0, 4.0}}, -2), RTensor({1.0, 2.0}));
  EXPECT_CEQ(min(RTensor{{1.0, 2.0}, {3.0, 4.0}}, -1), RTensor({1.0, 3.0}));
}

}  // namespace tensor_test
