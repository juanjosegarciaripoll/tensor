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
    // M(a1,a2,b1,b2) = Q'(a1,i,a2) P(b1,i,b2)
    t M = op?
      foldc(Q, 1, fold(*op, 1, P, 1), 0) :
      foldc(Q, 1, P, 1);
    // M(a1,a2,b1,b2) -> M(a1,b1,a2,b2)
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
      fold(fold(P, 1, *op, -1), 1, M0, 1) :
      // P(b0,i0,b1) M(a1,b1,a2,b2) -> M(b0,i0,a1,a2,b2)
      fold(P, -1, M0, 1);
    // Q'(a0,[i0,a1]) M(b0,[i0,a1],a2,b2) -> M(a0,b0,a2,b2)
    return foldc(reshape(Q, a0,i0*a1), -1, reshape(M, b0,i0*a1,a2,b2), 1);
  }

  template<class Tensor>
  static const Tensor prop_matrix_sub_qform(const Tensor &L, const Tensor &R)
  {
    index a1,a2,a3,b1,b2,b3;
    Tensor Q;
    //
    // We have some quadratic function expressed as
    //	E = L([a3,b3],a1,b1) P'(a1,i,a2) Op(i,j) P(b1,j,b2) R(a2,b2,[a3,b3])
    //	E = P'(i,a1,a2) Q([i,a1,a2],[j,b1,b2]) P(j,b1,b2)
    // where
    //	Q([a1,i,a2],[b1,j,b2]) = Op(i,j) (R*L)
    //
    // Notice that in L(a1,[a3,b3],b1), 'a1' is the index associated with the
    // complex conjugate of P, while in R(b2,[a3,b3],a2) it is 'a2', the last
    // one. IT IS CRITICAL THAT WE GET THE ORDER OF THE a's AND b's IN Q RIGHT.
    // Use effective_Hamiltonian_test(), by setting opts.debug=1 in
    // ground_state() to detect bugs in these functions.
    //
    if (L.is_empty()) {
	// P is the matrix corresponding to the first site, and thus there is no
	// "L" matrix or rather L(a1,b1,a3,b3) = delta(a1,a3)\delta(b1,b3).
	if (R.is_empty()) {
	    // This state has a single site.
	    a2 = b2 = a1 = b1 = 1;
	    Q = Tensor::eye(1);
	} else {
	    R.get_dimensions(&a2, &b2, &a1, &b1);
	    // R(a2,b2,a1,b1) -> R(b1,b2,a1,a2) -> R(a1,a2,b1,b2) =: Q
	    Q = transpose(reshape(permute(R, 0,3), b1*b2,a1*a2));
	}
    } else if (R.is_empty()) {
	// Similar as before, but P is the matrix is the one of the last site
	// and R(a2,b2,a3,b3) = delta(a2,a3)delta(b2,b3)
	L.get_dimensions(&a2, &b2, &a1, &b1);
	// L(a2,b2,a1,b1) -> L(b1,b2,a1,a2) -> L(a1*a2,b1*b2) =: Q
	Q = transpose(reshape(permute(L, 0,3), b1*b2,a1*a2));
    } else {
	L.get_dimensions(&a3, &b3, &a1, &b1);
	R.get_dimensions(&a2, &b2, &a3, &b3);
	// L(a3,b3,a1,b1)R(a2,b2,a3,b3) -> Q(a1,b1,a2,b2)
	Q = fold(reshape(L, a3*b3,a1,b1),0, reshape(R, a2,b2,a3*b3),2);
	// Q(a1,b1,a2,b2) -> Q(a1,a2,b1,b2)
	Q = reshape(permute(Q, 1,2), a1*a2,b1*b2);
    }
    return Q;
}



} //namespace mps
  
