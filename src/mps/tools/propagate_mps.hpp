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

#include <mps/tools.h>

namespace mps {

  template<class Tensor>
  static inline const Tensor
  do_propagate_right(Tensor v, const Tensor &A, const Tensor &op)
  {
    /* v(a,a') A(a',i,b) */
    v = fold(v, -1, A, 0);

    /* v(a,i,b) Op(i,j) -> v(a,j,b) */
    if (!op.is_empty()) {
	v = foldin(op, -1, v, 1);
    }

    /* v(a,i,b) A*(a,i,b') l(b') -> v(b,b') */

    tensor::index a,i,b;
    v.get_dimensions(&a,&i,&b);
    v = foldc(reshape(A, a*i,b), 0, reshape(v, a*i,b), 0);

    return v;
  }

}
