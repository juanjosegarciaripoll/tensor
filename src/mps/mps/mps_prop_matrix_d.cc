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
#include "mps_prop_matrix.cc"

namespace mps {

  const RTensor prop_matrix_close(const RTensor &N)
  {
    return do_prop_close(N);
  }

  const RTensor prop_matrix(const RTensor &M0, int sense, const RTensor &Q,
			    const RTensor &P, const RTensor *op)
  {
    if (M0.is_empty()) {
      return do_prop_init(Q, P, op);
    } else if (sense > 0) {
      return do_prop_right(M0, Q, P, op);
    } else {
      return do_prop_left(M0, Q, P, op);
    }
  }

} //namespace mps
  
