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

#include <algorithm>
#include <functional>
#include "loops.h"
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include <tensor/linalg.h>
#include "../src/linalg/find_blocks.hpp"

namespace tensor_test {

using namespace tensor;

template <class Tensor>
void test_find_blocks(const Tensor &A,
                      const std::vector<Indices> &expected_row_indices,
                      const std::vector<Indices> &expected_column_indices,
                      double tol = 0.0) {
  std::vector<Indices> rows, columns;
  bool separable = linalg::find_blocks(A, rows, columns, tol);

  if (expected_row_indices.size() == 0) {
    ASSERT_EQ(separable, false);
    return;
  }
#if 0
  for (index i = 0; i < rows.size(); ++i) {
    std::cerr << "row[" << i << "]=" << rows[i] << '\n'
              << "col[" << i << "]=" << columns[i] << '\n';
  }
#endif
  ASSERT_TRUE(rows.size() == expected_row_indices.size());
  ASSERT_TRUE(columns.size() == expected_column_indices.size());
  for (index i = 0; i < rows.size(); ++i) {
    ASSERT_TRUE(all_equal(rows[i], expected_row_indices[i]));
    ASSERT_TRUE(all_equal(columns[i], expected_column_indices[i]));
  }
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

TEST(RMatrixTest, FindBlocks001) {
  RTensor A{{1.0, 0.0}, {0.0, 1.0}};
  std::vector<Indices> rows = {{}, {0}, {1}};
  std::vector<Indices> cols = {{}, {0}, {1}};
  test_find_blocks(A, rows, cols);
}

TEST(RMatrixTest, FindsSingleBlock001) {
  RTensor A{{1.0, 3.0}, {2.0, 1.0}};
  std::vector<Indices> rows = {};
  std::vector<Indices> cols = {};
  test_find_blocks(A, rows, cols);
}

TEST(RMatrixTest, FindsSingleBlock002) {
  // clang-format off
  RTensor A{{0.0, 0.0, 3.0, 5.0},
            {0.0, 0.0, 1.0, 4.0},
            {2.0, 4.0, 0.0, 0.0},
            {1.0, 2.0, 0.0, 3.0}};
  // clang-format on
  std::vector<Indices> rows = {};
  std::vector<Indices> cols = {};
  test_find_blocks(A, rows, cols);
}

TEST(RMatrixTest, FindsSingleBlock003) {
  // clang-format off
  RTensor A{{1.0, 0.0, 3.0, 0.2},
            {0.0, 3.0, 0.0, 0.0},
            {2.0, 0.0, 3.0, 0.0},
            {0.0, 3.0, 0.0, 4.0}};
  // clang-format on
  std::vector<Indices> rows = {};
  std::vector<Indices> cols = {};
  test_find_blocks(A, rows, cols);
}

TEST(RMatrixTest, FindsTwoBlocks001) {
  // clang-format off
  RTensor A{{1.0, 3.0, 0.0, 0.0},
            {2.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, 3.0, 2.0},
            {0.0, 0.0, 1.0, 4.0}};
  // clang-format on
  std::vector<Indices> rows = {{}, {0, 1}, {2, 3}};
  std::vector<Indices> cols = {{}, {0, 1}, {2, 3}};
  test_find_blocks(A, rows, cols);
}

TEST(RMatrixTest, FindsTwoBlocks002) {
  // clang-format off
  RTensor A{{1.0, 0.0, 3.0, 0.0},
            {0.0, 3.0, 0.0, 4.0},
            {2.0, 0.0, 3.0, 0.0},
            {0.0, 2.0, 0.0, 4.0}};
  // clang-format on
  std::vector<Indices> rows = {{}, {0, 2}, {1, 3}};
  std::vector<Indices> cols = {{}, {0, 2}, {1, 3}};
  test_find_blocks(A, rows, cols);
}

TEST(RMatrixTest, FindsTwoBlocks003) {
  // clang-format off
  RTensor A{{0.0, 0.0, 3.0, 5.0},
            {0.0, 0.0, 1.0, 4.0},
            {2.0, 4.0, 0.0, 0.0},
            {1.0, 2.0, 0.0, 0.0}};
  // clang-format on
  std::vector<Indices> rows = {{}, {2, 3}, {0, 1}};
  std::vector<Indices> cols = {{}, {0, 1}, {2, 3}};
  test_find_blocks(A, rows, cols);
}

TEST(RMatrixTest, FindsThreeBlocks001) {
  // clang-format off
  RTensor A{{1.0, 0.0, 3.0, 0.0},
            {0.0, 3.0, 0.0, 0.0},
            {2.0, 0.0, 3.0, 0.0},
            {0.0, 0.0, 0.0, 4.0}};
  // clang-format on
  std::vector<Indices> rows = {{}, {0, 2}, {1}, {3}};
  std::vector<Indices> cols = {{}, {0, 2}, {1}, {3}};
  test_find_blocks(A, rows, cols);
}

TEST(RMatrixTest, FindsEmptyBlocks000) {
  RTensor A{{0.0, 0.0}, {0.0, 0.0}};
  std::vector<Indices> rows = {{0, 1}};
  std::vector<Indices> cols = {{0, 1}};
  test_find_blocks(A, rows, cols);
}

TEST(RMatrixTest, FindsEmptyBlocks001) {
  RTensor A{{1.0, 0.0}, {0.0, 0.0}};
  std::vector<Indices> rows = {{1}, {0}};
  std::vector<Indices> cols = {{1}, {0}};
  test_find_blocks(A, rows, cols);
}

TEST(RMatrixTest, FindsEmptyBlocks002) {
  RTensor A{{0.0, 0.0}, {0.0, 1.0}};
  std::vector<Indices> rows = {{0}, {1}};
  std::vector<Indices> cols = {{0}, {1}};
  test_find_blocks(A, rows, cols);
}

}  // namespace tensor_test
