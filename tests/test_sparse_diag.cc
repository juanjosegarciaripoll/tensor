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

#include <tensor/sparse.h>
#include "loops.h"
#include <gtest/gtest.h>

namespace tensor_test {

TEST(RSparseTest, MakeDiagonalMatrix1x1) {
  auto matrix = RSparse::diag(RTensor::eye(1,1), Indices{0}, 1, 1);
  EXPECT_EQ(matrix.rows(), 1);
  EXPECT_EQ(matrix.columns(), 1);
  EXPECT_EQ(matrix.length(), 1);
  EXPECT_ALL_EQUAL(full(matrix), RTensor::eye(1,1));
}

TEST(RSparseTest, MakeDiagonalMatrix2x2x1diag) {
  RTensor data{{1.0, 2.0}};
  {
	auto matrix = RSparse::diag(data, Indices{0}, 2, 2);
	auto expected = RTensor{{1.0, 0.0}, {0.0, 2.0}};
	EXPECT_EQ(matrix.length(), 2);
	EXPECT_ALL_EQUAL(full(matrix), expected);
  }
  {
	auto matrix = RSparse::diag(data, Indices{1}, 2, 2);
	auto expected = RTensor{{0.0, 1.0}, {0.0, 0.0}};
	EXPECT_EQ(matrix.length(), 1);
	EXPECT_ALL_EQUAL(full(matrix), expected);
  }
  {
	auto matrix = RSparse::diag(data, Indices{-1}, 2, 2);
	auto expected = RTensor{{0.0, 0.0}, {1.0, 0.0}};
	EXPECT_EQ(matrix.length(), 1);
	EXPECT_ALL_EQUAL(full(matrix), expected);
  }
  {
	auto matrix = RSparse::diag(data, Indices{2}, 2, 2);
	auto expected = RTensor{{0.0, 0.0}, {0.0, 0.0}};
	EXPECT_EQ(matrix.length(), 0);
	EXPECT_ALL_EQUAL(full(matrix), expected);
  }
}

TEST(RSparseTest, MakeDiagonalMatrixInteger) {
  RTensor data({1.0, 2.0});
  {
	auto matrix = RSparse::diag(data, 0, 2, 2);
	auto expected = RTensor{{1.0, 0.0}, {0.0, 2.0}};
	EXPECT_EQ(matrix.length(), 2);
	EXPECT_ALL_EQUAL(full(matrix), expected);
  }
  {
	auto matrix = RSparse::diag(data, 1, 2, 2);
	auto expected = RTensor{{0.0, 1.0}, {0.0, 0.0}};
	EXPECT_EQ(matrix.length(), 1);
	EXPECT_ALL_EQUAL(full(matrix), expected);
  }
  {
	auto matrix = RSparse::diag(data, -1, 2, 2);
	auto expected = RTensor{{0.0, 0.0}, {1.0, 0.0}};
	EXPECT_EQ(matrix.length(), 1);
	EXPECT_ALL_EQUAL(full(matrix), expected);
  }
  {
	auto matrix = RSparse::diag(data, 2, 2, 2);
	auto expected = RTensor{{0.0, 0.0}, {0.0, 0.0}};
	EXPECT_EQ(matrix.length(), 0);
	EXPECT_ALL_EQUAL(full(matrix), expected);
  }
}

TEST(RSparseTest, MakeDiagonalMatrix2x2x2diags) {
  {
	RTensor data{{1.0, 2.0}, {3.0, 4.0}};
	auto matrix = RSparse::diag(data, Indices{0, 1}, 2, 2);
	auto expected = RTensor{{1.0, 3.0}, {0.0, 2.0}};
	EXPECT_EQ(matrix.length(), 3);
	EXPECT_ALL_EQUAL(full(matrix), expected);
  }
  {
	RTensor data{{1.0, 2.0}, {3.0, 4.0}};
	auto matrix = RSparse::diag(data, Indices{0, -1}, 2, 2);
	auto expected = RTensor{{1.0, 0.0}, {3.0, 2.0}};
	EXPECT_EQ(matrix.length(), 3);
	EXPECT_ALL_EQUAL(full(matrix), expected);
  }
  {
	RTensor data{{1.0, 2.0}, {3.0, 4.0}};
	auto matrix = RSparse::diag(data, Indices{0, -2}, 2, 2);
	auto expected = RTensor{{1.0, 0.0}, {0.0, 2.0}};
	EXPECT_EQ(matrix.length(), 2);
	EXPECT_ALL_EQUAL(full(matrix), expected);
  }
  {
	RTensor data{{1.0, 2.0}, {3.0, 4.0}};
	auto matrix = RSparse::diag(data, Indices{-1, 1}, 2, 2);
	auto expected = RTensor{{0.0, 3.0}, {1.0, 0.0}};
	EXPECT_EQ(matrix.length(), 2);
	EXPECT_ALL_EQUAL(full(matrix), expected);
  }
}

TEST(RSparseTest, MakeDiagonalMatrixShapeDeduction) {
  RTensor data{1.0, 2.0};
  {
	auto matrix = RSparse::diag(data, 0);
	auto expected = RTensor{{1.0, 0.0}, {0.0, 2.0}};
	EXPECT_EQ(matrix.length(), 2);
	EXPECT_ALL_EQUAL(full(matrix), expected);
  }
  {
	auto matrix = RSparse::diag(data, 1);
	auto expected = RTensor{{0.0, 1.0, 0.0}, {0.0, 0.0, 2.0}};
	EXPECT_EQ(matrix.length(), 2);
	EXPECT_ALL_EQUAL(full(matrix), expected);
  }
  {
	auto matrix = RSparse::diag(data, -1);
	auto expected = RTensor{{0.0, 0.0}, {1.0, 0.0}, {0.0, 2.0}};
	EXPECT_EQ(matrix.length(), 2);
	EXPECT_ALL_EQUAL(full(matrix), expected);
  }
  {
	auto matrix = RSparse::diag(data, 2);
	auto expected = RTensor{{0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 2.0}};
	EXPECT_EQ(matrix.length(), 2);
	EXPECT_ALL_EQUAL(full(matrix), expected);
  }
}


}
