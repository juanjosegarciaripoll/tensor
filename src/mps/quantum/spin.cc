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
#include <tensor/gen.h>
#include <mps/quantum.h>

namespace mps {

  /*!\defgroup QM Quantum Mechanics
   */

  /**\f$\sigma_x\f$ Pauli matrix*/
  extern const RTensor Pauli_id(igen << 2 << 2, rgen << 1.0 << 0.0 << 0.0 << 1.0);
  extern const RTensor Pauli_x(igen << 2 << 2, rgen << 0.0 << 1.0 << 1.0 << 0.0);
  extern const RTensor Pauli_z(igen << 2 << 2, rgen << 1.0 << 0.0 << 0.0 << -1.0);
  extern const CTensor Pauli_y(igen << 2 << 2,
                               cgen << 0.0 << to_complex(0.0,1.0)
                               << to_complex(0.0,-1.0) << 0.0);

  /**Compute the angular momentum operators for a given total spin.
     \ingroup QM
  */

  void spin_operators(double s, CTensor *sx, CTensor *sy, CTensor *sz)
  {
    if (s < 0.5 || s > 3.0) {
      std::cerr << "spin_operators(): the spin value " << s << " is not valid\n";
      abort();
    }
    tensor::index d = (tensor::index)floor(2*s+1);

    RTensor aux1(d), aux2(d-1);
    for (tensor::index i = 0; i < d; i++) {
      double m = s - i;
      aux1.at(i) = m;
    }
    for (tensor::index i = 0; i < (d-1); i++) {
      double m = s - i;
      aux2.at(i) = sqrt(s*(s+1) - (m-1)*m);
    }
    RTensor sp = diag(aux2,+1);
    RTensor sm = diag(aux2,-1);
    *sx = to_complex(0.5 * (sp + sm));
    *sy = to_complex(0.0,0.5)* (sm - sp);
    *sz = to_complex(diag(aux1));
  }
}
