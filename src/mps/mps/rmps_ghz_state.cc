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
#include <mps/mps.h>

namespace mps {

  using namespace tensor;

  const RMPS ghz_state(index length, bool periodic)
  {
    RMPS output(length, 2, 2, periodic);
    if (length == 1) {
      double v = 1.0/sqrt(2.0);
      RTensor &aux = output.at(0);
      aux.fill_with_zeros();
      if (periodic) {
	aux.at(0,1,0) = v;
	aux.at(1,1,1) = v;
      } else {
	aux.at(0,0,0) = v;
	aux.at(0,1,0) = v;
      }
    } else {
      double v = 1.0/sqrt(sqrt(2.0));
      for (size_t i = 0; i < length; i++) {
	RTensor &aux = output.at(i);
	aux.fill_with_zeros();
	if (i == 0) {
	  aux.at(0,0,0) = v;
	  aux.at((periodic?1:0),1,1) = v;
	} else if (i == (length-1)) {
	  aux.at(0,0,0) = v;
	  aux.at(1,1,(periodic?1:0)) = v;
	} else {
	  aux.at(0,0,0) = 1.0;
	  aux.at(1,1,1) = 1.0;
	}
      }
    }
    return output;
  }

} // namespace mps
