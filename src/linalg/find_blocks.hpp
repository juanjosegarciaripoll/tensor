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

#include <vector>
#include <algorithm>
#include <tensor/flags.h>
#include <tensor/tensor.h>

namespace linalg {

using namespace tensor;
using tensor::index;

/*Find blocks in a block-diagonal matrix.*/
/*If A is a block diagonal matrix, that means that, after appropiate reordering
    of columns and rows, the nonzero elements of the matrix form (rectangular)
    boxes distributed along the diagonal. For instance take
\code
	A = {0, 0, 1, 1,
	     0, 0, 1, 1,
	     2, 2, 0, 0,
	     2, 2, 0, 0};
\endcode
    This matrix can be reordered as
\code
	A2= {1, 1, 0, 0,
	     1, 1, 0, 0,
	     0, 0, 2, 2,
	     0, 0, 2, 2};
\endcode
    which shows the evident block-diagonal structure. The routine find_block()
    takes as input a matrix such as A and produces a two lists of vectors, each
    one denoting the rows and columns of the nonzero blocks in the matrix.
  */
template <class Tensor>
bool find_blocks(const Tensor &A, std::vector<Indices> &row_indices,
                 std::vector<Indices> &column_indices, double tol = 0.0) {
  index N = A.rows();
  index M = A.columns();
  if (N > M) {
    return find_blocks(transpose(A), column_indices, row_indices);
  }

  const index aux = 0;
  const index empty = ~aux;
  std::vector<index> row_block(N, empty);
  std::vector<index> column_block;
  column_block.reserve(M);

  index nblocks = 0;
  auto data = A.begin_const();
  for (index col = 0; col < M; col++) {
    // For each col, we see what rowumns it is linked to. If some of these
    // rowumns belongs to a block, and the current block is not set, we
    // set our current block. Otherwise, we mix different blocks.
    index curr_block = col;
    bool set = false;
    for (index row = 0; row < N; row++, ++data) {
      bool significant = abs(real(*data)) + abs(imag(*data)) > tol;
      if (significant) {
        set = true;
        index new_block = row_block[row];
        if (new_block == empty) {
          row_block[row] = curr_block;
        } else if (new_block != curr_block) {
          nblocks--;
          std::replace(std::begin(row_block), std::end(row_block), curr_block,
                       new_block);
          std::replace(std::begin(column_block), std::end(column_block),
                       curr_block, new_block);
          curr_block = new_block;
        }
      }
    }
    if (set) {
      nblocks++;
      column_block.push_back(curr_block);
    } else {
      column_block.push_back(empty);
    }
  }
  if (tensor::FLAGS.get(tensor::TENSOR_DEBUG_BLOCK_SVD)) {
    std::cout << "*** find_blocks: nxm=" << N << "x" << M
              << ", n_blocks=" << nblocks << std::endl;
  }
  if (nblocks <= 1) {
    return false;
  }

  std::vector<index> buffer;
  buffer.reserve(std::max<index>(N, M));
  column_indices.reserve(nblocks);
  row_indices.reserve(nblocks);

  auto extract_positions = [&](std::vector<index> &v, index start,
                               index which) {
    buffer.clear();
    index count = 0;
    for (index ndx = start; ndx < static_cast<index>(v.size()); ++ndx) {
      auto &x = v[ndx];
      if (x == which) {
        x = empty;
        ++count;
        buffer.push_back(ndx);
      }
    }
    assert(count);
    Indices output = Indices::empty(count);
    std::copy(std::begin(buffer), std::end(buffer), std::begin(output));
    return output;
  };

  for (index start = 0; start < static_cast<index>(column_block.size());
       ++start) {
    auto block = column_block[start];
    if (block != empty) {
      column_indices.push_back(extract_positions(column_block, start, block));
      row_indices.push_back(extract_positions(row_block, 0, block));
    }
  }
  assert(ssize(column_indices) == nblocks);
  assert(ssize(row_indices) == nblocks);
  return true;
}

}  // namespace linalg
