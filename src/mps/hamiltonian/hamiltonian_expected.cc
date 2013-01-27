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

#include <mps/hamiltonian.h>

namespace mps {

  template<class MPS>
  static inline double do_expected(const MPS &P, const Hamiltonian &theH, double t)
  {
    CTensor O1, O2, H;
    cdouble E = number_zero<cdouble>();
    index N = theH.size();
    for (index k = 0, k2 = 1; k < N; k++, k2++) {
	H = theH.local_term(k, t);
	if (!H.is_empty()) {
	    E = E + expected(P, H, k);
	}
	if (k2 == N) {
	    if (k == 0 || !theH.is_periodic())
		break;
	    k2 = 0;
	}
	if (N > 0) {
	    index depth = theH.interaction_depth(k, t);
	    for (index i = 0; i < depth; i++) {
		E += expected(P, theH.interaction_left(k, i, t), k,
			      theH.interaction_right(k, i, t), k2);
	    }
	}
    }
#if 0
    if (abs(im_part(E))/max(abs(E),1.0) > 1e-10) {
	std::cerr << "In Hamiltonian::expected_value(), got complex results when computing\n"
	    "the energy: " << re_part(E) << "+" << im_part(E) << "i\n";
	//myabort();
    }
#endif
    return real(E);
  }

} // namespace mps

