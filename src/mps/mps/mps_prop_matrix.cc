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

namespace mps {

  template<typename t>
  static inline const t do_prop_init(const t &Q, const t &P, const t *op)
  {
    t M = op?
      foldc(Q, 1, fold(*op, 1, P, 1), 0) :
      foldc(Q, 1, P, 1);
    return permute(M, 1, 2);
  }

  template<typename t>
  static inline const t do_prop_close(const t &M)
  {
    index a1, b1, a2, b2;
    M.get_dimensions(&a1, &b1, &a2, &b2);
    return trace(reshape(M, a1*b1, a2*b2), 0, -1);
  }

  template<typename t>
  static inline const t do_prop_right(const t &M0, const t &Q, const t &P, const t *op)
  {
    index a1,b1,a2,b2,i2,a3,b3;
    M0.get_dimensions(&a1, &b1, &a2, &b2);
    Q.get_dimensions(&a2, &i2, &a3);
    P.get_dimensions(&b2, &i2, &b3);

    t M = op ?
      // M(a1,b1,a2,b2) Op(j2,i2) Q'(a2,j2,a3) -> M(a1,b1,b2,i2,a3)
      fold(M0, 2, fold(*op, 0, conj(Q), 1), 1) :
      // M(a1,b1,a2,b2) Q'(a2,i2,a3) -> M(a1,b1,b2,i2,a3)
      fold(M0, 2, conj(Q), 0);
    // M(a1,b1,[b2,i2],a3) P([b2,i2],b3) -> M(a1,b1,a3,b3)
    return fold(reshape(M, a1,b1,b2*i2,a3), 2, reshape(P, b2*i2,b3), 0);
  }

  template<typename t>
  static inline const t do_prop_left(const t &M0, const t &Q, const t &P, const t *op)
  {
    index a1,b1,a2,b2,a0,b0,i0;
    M0.get_dimensions(&a1,&b1,&a2,&b2);
    Q.get_dimensions(&a0,&i0,&a1);
    P.get_dimensions(&b0,&i0,&b1);
    t M = op?
      // P(b0,j0,b1) Op(i0,j0) M(a1,b1,a2,b2) -> M(b0,i0,a1,a2,b2)
      M = fold(fold(P, 1, *op, -1), 1, M0, 1) :
      // P(b0,i0,b1) M(a1,b1,a2,b2) -> M(b0,i0,a1,a2,b2)
      M = fold(P, -1, M0, 1);
    // Q'(a0,[i0,a1]) M(b0,[i0,a1],a2,b2) -> M(a0,b0,a2,b2)
    return foldc(reshape(Q, a0,i0*a1), -1, reshape(M, b0,i0*a1,a2,b2), 1);
  }

} //namespace mps
  
