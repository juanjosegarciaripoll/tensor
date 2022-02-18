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
  index minrc = std::min(rows, cols);

  std::vector<Indices> row_indices, column_indices;
  if (!find_blocks<Tensor>(A, row_indices, column_indices)) {
    return svd(A, pU, pVT, economic);
  }
  index nblocks = row_indices.size();
  RTensor s = RTensor::zeros(minrc);
  if (pU) {
    *pU = Tensor::zeros(rows, economic ? minrc : rows);
  }
  if (pVT) {
    *pVT = Tensor::zeros(economic ? minrc : cols, cols);
  }

  RTensor stemp;
  Tensor Utemp, Vtemp;
  Tensor *pUtemp = pU ? &Utemp : 0;
  Tensor *pVtemp = pVT ? &Vtemp : 0;
  for (index b = 0, sndx = 0; b < nblocks; b++) {
    Tensor m = A(range(row_indices[b]), range(column_indices[b]));
    if (m.size() > 1) {
      stemp = svd(m, pUtemp, pVtemp, economic);
      index slast = sndx + stemp.ssize() - 1;
      s.at(range(sndx, slast)) = stemp;
      if (pU) {
        (*pU).at(range(row_indices[b]), range(sndx, slast)) = Utemp;
      }
      if (pVT) {
        (*pVT).at(range(sndx, slast), range(column_indices[b])) = Vtemp;
      }
      sndx = slast + 1;
    } else {
      index row = row_indices[b][0];
      index col = column_indices[b][0];
      double aux = abs(m[0]);
      s.at(sndx) = aux;
      if (pU) {
        (*pU).at(row, sndx) = 1.0;
      }
      if (pVT) {
        (*pVT).at(sndx, col) = m[0] / aux;
      }
      ++sndx;
    }
  }

  Indices ndx = sort_indices(s, true);
  s = s(range(ndx));
  if (pU) *pU = (*pU)(_, range(ndx));
  if (pVT) *pVT = (*pVT)(range(ndx), _);
  return s;
}

}  // namespace linalg
