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

#include <tensor/indices.h>
#include <mps/quantum.h>
#include <tensor/io.h>

namespace mps {

  using tensor::index;

  /**Return the Fock number operator truncated for a space of up to 'nmax' bosons.
     \ingroup QM
  */
  RSparse number_operator(int nmax)
  {
    index d = nmax+1; // Matrix size
    Indices ndx = iota(0, nmax);
    RTensor n = linspace(0, nmax, nmax+1);
    return RSparse(ndx, ndx, n, d, d);
  }

  /**Return the Fock destruction operator truncated for a space of up to 'nmax' bosons.
     \ingroup QM
  */
  RSparse destruction_operator(int nmax)
  {
    index d = nmax+1; // Matrix size
    Indices row = iota(0, nmax-1);
    Indices col = iota(1, nmax);
    RTensor n = sqrt(linspace(1.0, nmax, nmax));
    return RSparse(row, col, n, d, d);
  }

  /**Return the Fock creation operator truncated for a space of up to 'nmax' bosons.
     \ingroup QM
  */
  RSparse creation_operator(int nmax)
  {
    index d = nmax+1; // Matrix size
    Indices row = iota(1, nmax);
    Indices col = iota(0, nmax-1);
    RTensor n = sqrt(linspace(1.0, nmax, nmax));
    return RSparse(row, col, n, d, d);
  }

  /**Return the wavefunction of a coherent state.
     \ingroup QM
  */
  RTensor coherent_state(double alpha, int nmax)
  {
    RTensor output(nmax+1);
    double c = exp(-alpha*alpha/2.0);
    for (int i = 0; i <= nmax; ) {
      output.at(i) = c;
      c = c * alpha / sqrt((double)(++i));
    }
    return output;
  }

  /**Return the wavefunction of a coherent state.
     \ingroup QM
  */
  CTensor coherent_state(cdouble alpha, int nmax)
  {
    CTensor output(nmax+1);
    double a2 = abs(alpha);
    cdouble c = exp(-a2*a2/2.0);
    for (int i = 0; i <= nmax; ) {
      output.at(i) = c;
      c = c * alpha / sqrt((double)(++i));
    }
    return output;
  }

}
