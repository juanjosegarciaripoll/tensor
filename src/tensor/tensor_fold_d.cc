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

#include "tensor_fold.cc"

namespace tensor {

/**Contraction of two tensors. A contraction is a generalization of the matrix
     product that we all know. The code \c C=fold(A,1,B,0) contracts tensors A
     and B, according to the second and first index as given by the formula
     \f[
     C_{i_0i_1i_3i_4} = \sum_j A_{i_0ji_1} B_{ji_3i_4}
     \f]

     There exist variants for these functions for contracting real with real,
     complex with complex, and real with complex tensors. Depending on the
     choice of indices, NDX1 and NDX2, the multiplication will be faster or
     slower. For instance, at least fast products are achieved via the ATLAS
     library when NDX=0 or 1.

     \ingroup Tensors
  */
RTensor fold(const RTensor &a, int ndx1, const RTensor &b, int ndx2) {
  RTensor output;
  do_fold<double, false>(output, a, ndx1, b, ndx2);
  return output;
}

/**Contraction of two tensors. The code \c C=foldc(A,n,B,m) acting on real
     tensors does the same as \c fold(A,n,B,m).

     \ingroup Tensors
  */
RTensor foldc(const RTensor &a, int ndx1, const RTensor &b, int ndx2) {
  RTensor output;
  do_fold<double, false>(output, a, ndx1, b, ndx2);
  return output;
}

void fold_into(RTensor &c, const RTensor &a, int ndx1, const RTensor &b,
               int ndx2) {
  do_fold<double, false>(c, a, ndx1, b, ndx2);
}

/**Matrix multiplication. \c mmult(A,B) is equivalent to \c fold(A,-1,B,0). */

RTensor mmult(const RTensor &m1, const RTensor &m2) {
  return fold(m1, -1, m2, 0);
}

void mmult_into(RTensor &c, const RTensor &m1, const RTensor &m2) {
  fold_into(c, m1, -1, m2, 0);
}

}  // namespace tensor
