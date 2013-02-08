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

#include <mps/itebd.h>

namespace mps {

  const RiTEBD infinite_aklt_state()
  {
    // From "The density-matrix renormalization group in
    // the age of matrix product states",
    // Ulrich Schollwoeck, Annals of Physics 326, 96 (2011)
    RTensor A = RTensor::zeros(igen << 2 << 3 << 2);
    // A(,0,) = sqrt(2/3) \sigma^+
    A.at(0,0,1) = sqrt(2.0/3.0);
    // A(,1,) = Pauli_z
    A.at(0,1,0) = -sqrt(1.0/3.0);
    A.at(1,1,1) = sqrt(1.0/3.0);
    // A(,2,) = sqrt(2/3) \sigma^-
    A.at(1,2,0) = -sqrt(2.0/3.0);
    RTensor lA = RTensor::ones(igen << 2);
    return RiTEBD(A, lA, A, lA);
  }

}
