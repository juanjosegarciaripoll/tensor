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

  template<class Tensor>
  iTEBD<Tensor>::iTEBD(const Tensor &newA) {
    assert(newA.rank() == 1);
    A = reshape(newA, igen << 1 << newA.dimension(0) << 1);
    A = A / norm2(A);
    lA = Tensor(igen << 1);
    lA.at(0) = 1.0;
    B = A;
    lB = lA;
  }

  template<class Tensor>
  iTEBD<Tensor>::iTEBD(const Tensor &newA, const Tensor &newB) {
    assert(newA.rank() == 1);
    assert(newB.rank() == 1);
    A = reshape(newA, igen << 1 << newA.dimension(0) << 1);
    A = A / norm2(A);
    lA = Tensor(igen << 1);
    lA.at(0) = 1.0;
    B = reshape(newB, igen << 1 << newB.dimension(0) << 1);
    B = B / norm2(B);
    lB = lA;
  }

  template<class Tensor>
  iTEBD<Tensor>::iTEBD(const Tensor &newA, const Tensor &newlA,
		       const Tensor &newB, const Tensor &newlB) :
    A(newA), lA(newlA), B(newB), lB(newlB)
  {
    assert(A.rank() == 3);
    assert(A.dimension(0) == lB.dimension(0));
    assert(A.dimension(2) == lA.dimension(0));
    assert(B.dimension(0) == lA.dimension(0));
    assert(B.dimension(2) == lB.dimension(0));
  }

}
