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
  static void set_canonical_2_sites_inner(MPS &P, const Tensor &Pij, index site,
					  int sense, index Dmax, double tol)
  {
    /*
     * Since the projector that we obtained spans two sites, we have to split
     * it, ensuring that we remain below the desired dimension Dmax.
     */
    index a1, i1, j1, c1;
    Pij.get_dimensions(&a1, &i1, &j1, &c1);
    Tensor Pi, Pj;
    RTensor s = linalg::svd(reshape(Pij, a1*i1,j1*c1), &Pi, &Pj, SVD_ECONOMIC);
    double factor;
    index b1 = where_to_truncate(s, tol, Dmax);
    if (b1 != s.size()) {
	Pi = change_dimension(Pi, -1, b1);
	Pj = change_dimension(Pj, 0, b1);
	s = change_dimension(s, 0, b1);
    }
    Pi = reshape(Pi, a1,i1,b1);
    Pj = reshape(Pj, b1,j1,c1);
    if (sense > 0) {
      P.at(site) = Pi;
      scale_inplace(Pj, 0, s);
      set_canonical(P, site+1, Pj, sense, true);
    } else {
      P.at(site+1) = Pj;
      scale_inplace(Pi,-1, s);
      set_canonical(P, site, Pi, sense, true);
    }
  }

} // namespace mps
