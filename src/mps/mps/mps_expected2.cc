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

#include <mps/mps.h>
#include <mps/mps_algorithms.h>

namespace mps {

  using namespace tensor;

  /* TWO-SITE CORRELATION FUNCTION */

  template <class MPS, class Tensor>
  typename Tensor::elt_t two_sites_expected(const MPS &a, const Tensor &Op1, size_t k1,
					   const Tensor &Op2, size_t k2)
  {
    if (k1 == k2) {
      return expected(a, mmult(Op1, Op2), k1);
    } else {
      Tensor M;
      const Tensor *op;
      k1 = a.normal_index(k1);
      k2 = a.normal_index(k2);
      for (index k = 0; k < a.size(); k++) {
	Tensor Pk = a[k];
	if (k == k1) {
	  op = &Op1;
	} else if (k == k2) {
	  op = &Op2;
	} else {
	  op = NULL;
	}
	M = prop_matrix(M, +1, Pk, Pk, op);
      }
      prop_close(M);
      return M[0];
    }
  }

} // namespace mps
