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
#include <tensor/tensor.h>
#include <tensor/io.h>
#include <tensor/linalg.h>
#include "find_blocks.hpp"

namespace linalg {

using tensor::index;

template <class Tensor>
RTensor do_block_svd(const Tensor &A, Tensor *pU, Tensor *pVT, bool economic) {
  index rows = A.rows();
  index cols = A.columns();
  if (rows != cols && !economic) return svd(A, pU, pVT, economic);

  std::vector<Indices> row_indices, column_indices;
  if (!find_blocks<Tensor>(A, row_indices, column_indices)) {
    return svd(A, pU, pVT, economic);
  }
  index L = std::min(rows, cols);
  RTensor s = RTensor::zeros(L);
  if (pU) {
    *pU = Tensor::zeros(rows, economic ? L : rows);
  }
  if (pVT) {
    *pVT = Tensor::zeros(economic ? L : cols, cols);
  }

  RTensor stemp;
  Tensor Utemp, Vtemp;
  Tensor *pUtemp = pU ? &Utemp : nullptr;
  Tensor *pVtemp = pVT ? &Vtemp : nullptr;
  for (index nblocks = ssize(row_indices), b = 0, sndx = 0; b < nblocks; b++) {
    const auto &block_rows = row_indices[static_cast<size_t>(b)];
    auto M = block_rows.ssize();
    const auto &block_columns = column_indices[static_cast<size_t>(b)];
    auto N = block_columns.ssize();
    if (M != 0 && N != 0) {
      L = std::min(M, N);
      // The first indices are for the empty rows and columns. We do not need
      // to compute the SVD in this case
      if (b == 0) {
        stemp = RTensor::zeros(L);
        Utemp = Tensor::eye(M, economic ? L : M);
        Vtemp = Tensor::eye(economic ? L : N, N);
      } else {
        stemp = svd(A(range(block_rows), range(block_columns)), pUtemp, pVtemp,
                    economic);
      }
      index slast = sndx + L - 1;
      s.at(range(sndx, slast)) = stemp;
      if (pU) {
        (*pU).at(range(block_rows), range(sndx, slast)) = Utemp;
      }
      if (pVT) {
        (*pVT).at(range(sndx, slast), range(block_columns)) = Vtemp;
      }
      sndx = slast + 1;
    }
  }
  Indices ndx = stable_sort_indices(s, true);
  s = s(range(ndx));
  if (pU) *pU = (*pU)(_, range(ndx));
  if (pVT) *pVT = (*pVT)(range(ndx), _);
  return s;
}

}  // namespace linalg
