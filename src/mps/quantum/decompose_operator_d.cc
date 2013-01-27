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

#include <mps/quantum.h>
#include "decompose_operator.cc"

namespace mps {

  /**Schmidt decomposition of a real operator. This function assumes that the
     operator \c U acts on two physical systems of the same dimension \c D, and
     finds out the decomposition
     \f[U = \sum_{k=1}^n O^{(1)}_k \otimes O^{(2)}_k\f]

     In practice \c U is a matrix \c D^2xD^2, and then O1 and O2 are
     three-dimensional tensors with dimensions \c DxDxn, where \c n is the total
     number of operators required in the previous decomposition.

     This decomposition is mostly useful for computing expected values using
     matrix product states.

     \ingroup QM
  */
  void
  decompose_operator(const RTensor &U, RTensor *O1, RTensor *O2) {
    do_decompose_operator(U, O1, O2);
  }


} // namespace mps
