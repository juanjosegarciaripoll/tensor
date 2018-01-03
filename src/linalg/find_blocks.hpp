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

#include <list>
#include <algorithm>
#include <tensor/flags.h>
#include <tensor/tensor.h>

namespace linalg {

  using namespace tensor;
  using tensor::index;

  static void
  replace_integer(index N, index v[], index old_value, index new_value)
  {
    for (index k = 0; k < N; k++) {
      if (v[k] == old_value) v[k] = new_value;
    }
  }

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
  template<class Tensor>
  bool
  find_blocks(const Tensor &A, index *pnblocks, Indices **pblock_rows, Indices **pblock_cols,
	      double tol = 0.0)
  {
    index N = A.rows();
    index M = A.columns();
    if (N > M)
      return find_blocks(transpose(A), pnblocks, pblock_cols, pblock_rows);

    index &nblocks = *pnblocks;
    Indices *&block_rows = *pblock_rows;
    Indices *&block_cols = *pblock_cols;

    index *row_block = new index[N];
    index *col_block = new index[M];
    const index aux = 0;
    const index empty = ~aux;

    for (index row = 0; row < N; row++)
      row_block[row] = empty;

    nblocks = 0;
    const typename Tensor::elt_t *data = A.begin_const();

    for (index col = 0; col < M; col++) {
      // For each col, we see what rowumns it is linked to. If some of these
      // rowumns belongs to a block, and the current block is not set, we
      // set our current block. Otherwise, we mix different blocks.
      index curr_block = col;
      bool set = 0;
      for (index row = 0; row < N; row++, data++) {
	bool significant = abs(real(*data)) + abs(imag(*data)) > tol;
	if (significant) {
	  set = 1;
	  index new_block = row_block[row];
	  if (new_block == empty) {
	    row_block[row] = curr_block;
	  } else if (new_block != curr_block) {
	    nblocks--;
	    replace_integer(N, row_block, curr_block, new_block);
	    replace_integer(col, col_block, curr_block, new_block);
	    curr_block = new_block;
	  }
	}
      }
      if (set) {
	nblocks++;
	col_block[col] = curr_block;
      } else {
	col_block[col] = empty;
      }
    }
    if (tensor::FLAGS.get(tensor::TENSOR_DEBUG_BLOCK_SVD)) {
      std::cout << "*** find_blocks: nxm=" << N << "x" << M
                << ", n_blocks=" << nblocks << std::endl;
    }
    if (nblocks == 1) {
      block_rows = 0;
      block_cols = 0;
      delete[] row_block;
      delete[] col_block;
      return false;
    }

    index *buffer = new index[std::max<index>(N,M)];
    block_cols = new Indices[nblocks];
    block_rows = new Indices[nblocks];

    for (index b = 0, col = 0; col < M; col++) {
      index block = col_block[col];
      if (block != empty) {
	index block_ncols = 0;
	for (index c = 0; c < M; c++) {
	  if (col_block[c] == block) {
	    buffer[block_ncols++] = c;
	    col_block[c] = empty;
	  }
	}
	assert(block_ncols);
	block_cols[b] = Indices(block_ncols);
	memcpy(block_cols[b].begin(), buffer, sizeof(index)*block_ncols);

	index block_nrows = 0;
	for (index r = 0; r < N; r++){
	  if (row_block[r] == block) {
	    buffer[block_nrows++] = r;
	  }
	}
	assert(block_nrows);
	block_rows[b] = Indices(block_nrows);
	memcpy(block_rows[b].begin(), buffer, sizeof(index)*block_nrows);
	b++;
      }
    }
    delete[] buffer;
    delete[] row_block;
    delete[] col_block;
    return true;
  }

} // namespace linalg
