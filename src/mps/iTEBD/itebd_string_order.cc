// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2012 Juan Jose Garcia Ripoll

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

#include <mps/tools.h>
#include <mps/itebd.h>

namespace mps {

  template<class Tensor>
  static inline const typename Tensor::elt_t
  do_string_order(const iTEBD<Tensor> &psi,
		  const Tensor &Opi, int i, const Tensor &Opmiddle,
		  const Tensor &Opj, int j)
  {
    if (i > j) {
      return do_string_order(psi, Opj, j, Opmiddle, Opi, i);
    } else {
      assert(psi.is_canonical());
      int site = i;
      Tensor v1 = psi.left_boundary(site);
      Tensor v2 = v1;
      v1 = propagate_right(v1, psi.combined_matrix(site), Opi);
      v2 = propagate_right(v2, psi.combined_matrix(site));
      ++site;
      while (site < j) {
	v1 = propagate_right(v1, psi.combined_matrix(site), Opmiddle);
	v2 = propagate_right(v2, psi.combined_matrix(site));
	++site;
      }
      return trace(propagate_right(v1, psi.combined_matrix(site), Opj)) /
	trace(propagate_right(v2, psi.combined_matrix(site)));
    }
  }

}
