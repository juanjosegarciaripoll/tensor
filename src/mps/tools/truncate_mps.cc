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

#include <mps/mps_algorithms.h>

namespace mps {

  template<class MPS>
  static const Indices
  expected_dimensions(const MPS &P, index Dmax, bool periodic)
  {
    index l = P.size();
    Indices d(l+1);
    if (periodic) {
      for (index i = 0; i <= l; i++)
	d.at(i) = Dmax;
    } else {
      d.at(0) = 1;
      d.at(l) = 1;
      for (index i = 0, c = 1; i < l; i++) {
	c *= P[i].dimension(1);
	if (c > Dmax) c = Dmax;
	d.at(i) = c;
      }
      for (index i = l, c = 1; i--;) {
	c *= P[i].dimension(1);
	if (c > Dmax) c = Dmax;
	if (c < d[i]) d.at(i) = c;
      }
    }
    return d;
  }

  template<class MPS>
  bool
  truncate_inner(MPS *Q, const MPS &P, index Dmax, bool periodic)
  {
    Indices d = expected_dimensions(P, Dmax, periodic);
    bool truncated = 0;
    index L = P.size();
    *Q = MPS(L);
    for (index k = 0; k < L; k++) {
      typename MPS::elt_t Qk = P[k];
      if (Qk.dimension(0) > d[k]) {
	truncated = 1;
	Qk = change_dimension(Qk, 0, d[k]);
      }
      if (Qk.dimension(2) > d[k+1]) {
	truncated = 1;
	Qk = change_dimension(Qk, 2, d[k+1]);
      }
      Q->at(k) = Qk;
    }
    return truncated;
  }

}
