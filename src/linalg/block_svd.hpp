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

#include <tensor/tensor.h>
#include <tensor/linalg.h>
#include "find_blocks.hpp"

namespace linalg {

  using tensor::index;

  template<class Tensor>
  RTensor
  do_block_svd(const Tensor &A, Tensor *pU, Tensor *pVT, bool economic)
  {
    index rows = A.rows();
    index cols = A.columns();
    index minrc = std::min(rows, cols);

    index nblocks;
    Indices *block_rows, *block_cols;
    if (!find_blocks<Tensor>(A, &nblocks, &block_rows, &block_cols)) {
      return svd(A, pU, pVT, economic);
    }

    if ((nblocks == 1) &&
	(block_rows[0].size() >= rows/2) &&
	(block_cols[0].size() >= cols/2)) {
      RTensor s = svd(A, pU, pVT, economic);
      delete[] block_rows;
      delete[] block_cols;
      return s;
    }

    RTensor s(minrc);
    s.fill_with_zeros();
    if (pU) {
      *pU = Tensor(rows, economic? minrc : rows);
      pU->fill_with_zeros();
    }
    if (pVT) {
      *pVT = Tensor(economic? minrc : cols, cols);
      pVT->fill_with_zeros();
    }

    RTensor stemp;
    Tensor Utemp, Vtemp;
    for (index b = 0, outrow = 0; b < nblocks; b++) {
      Tensor m = A(range(block_rows[b]), range(block_cols[b]));
      if (m.size() > 1) {
	stemp = svd(m, (pU? &Utemp : 0), (pVT? &Vtemp : 0), economic);
	s.at(range(outrow, outrow + stemp.size() - 1)) = stemp;
	if (pU) {
	  (*pU).at(range(block_rows[b]), range(outrow, outrow + stemp.size() - 1)) = Utemp;
	}
	if (pVT) {
	  (*pVT).at(range(outrow, outrow + stemp.size() - 1), range(block_cols[b])) = Vtemp;
	}
	outrow += stemp.size();
      } else {
	index row = block_rows[b][0];
	index col = block_cols[b][0];
	double aux = abs(m[0]);
	s.at(outrow) = aux;
	if (pU) {
	  (*pU).at(row,outrow) = 1.0;
	}
	if (pVT) {
	  (*pVT).at(outrow,col) = m[0]/aux;
	}
	outrow++;
      }
    }
    delete[] block_rows;
    delete[] block_cols;

    Indices ndx = sort_indices(s, true);
    s = s(range(ndx));
    if (pU) *pU = (*pU)(range(), range(ndx));
    if (pVT) *pVT = (*pVT)(range(ndx), range());

    return s;
  }


} // namespace linalg
