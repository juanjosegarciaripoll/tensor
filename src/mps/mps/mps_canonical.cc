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

#include <algorithm>
#include <tensor/linalg.h>
#include <mps/mps.h>

namespace mps {

  template<class MPS, class Tensor>
  static void set_canonical_inner(MPS &psi, index ndx, const Tensor &t,
				  int sense, bool truncate)
  {
    index b1, i1, b2;
    if (sense == 0) {
      std::cerr << "In MPS::set_canonical(), " << sense
		<< " is not a valid direction.";
      abort();
    } else if (sense > 0) {
      if (ndx+1 == psi.size()) {
	psi.at(ndx) = t;
      } else {
	Tensor U, V;
	t.get_dimensions(&b1, &i1, &b2);
	RTensor s = linalg::svd(reshape(t, b1*i1, b2), &U, &V, SVD_ECONOMIC);
	index l = s.size();
	index new_l = truncate?
	  where_to_truncate(s, -1.0, l) :
	  std::min<index>(b1*i1,b2);
	if (new_l != l) {
	  U = change_dimension(U, 1, new_l);
	  V = change_dimension(V, 0, new_l);
	  s = change_dimension(s, 0, new_l);
	  s = s / norm2(s);
	  l = new_l;
	}
	psi.at(ndx) = reshape(U, b1,i1,l);
	scale_inplace(V, 0, s);
	psi.at(ndx+1) = fold(V, -1, psi[ndx+1], 0);
      }
    } else {
      if (ndx == 0) {
	psi.at(ndx) = t;
      } else {
	Tensor U, V;
	t.get_dimensions(&b1, &i1, &b2);
	RTensor s = linalg::svd(reshape(t, b1, i1*b2), &V, &U, SVD_ECONOMIC);
	index l = s.size();
	index new_l = truncate?
	  where_to_truncate(s, -1.0, l) :
	  std::min<index>(b1,i1*b2);
	if (new_l != l) {
	  U = change_dimension(U, 0, new_l);
	  V = change_dimension(V, 1, new_l);
	  s = change_dimension(s, 0, new_l);
	  s = s / norm2(s);
	  l = new_l;
	}
	psi.at(ndx) = reshape(U, l,i1,b2);
	scale_inplace(V, -1, s);
	psi.at(ndx-1) = fold(psi[ndx-1], -1, V, 0);
      }
    }
  }

  template<class MPS>
  static const MPS canonical_form_inner(const MPS &psi, int sense)
  {
    MPS output(psi);
    if (sense < 0) {
      for (index i = psi.size(); i; ) {
	--i;
	set_canonical(output, i, output[i], sense);
      }
    } else {
      for (index i = 0; i < psi.size(); i++) {
	set_canonical(output, i, output[i], sense);
      }
    }
    return output;
  }


} // namespace mps
