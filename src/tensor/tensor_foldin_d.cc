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

#include "tensor_foldin.cc"

namespace tensor {

/**Contraction of two tensors with complex conjugation. Similar to the fold()
     routine, but the indices of the first tensor are inserted in the output.
     In other words the code \c
     C=foldc(A,1,B,0) performs
     \f[
     C_{i_3i_0i_1i_4} = \sum_j A_{i_0ji_1}^\star B_{ji_3i_4}
     \f]

     \ingroup Tensors
  */
const Tensor<double> foldin(const Tensor<double> &a, int _ndx1,
                            const Tensor<double> &b, int _ndx2) {
  Tensor<double> output;
  do_foldin_into(output, a, _ndx1, b, _ndx2);
  return output;
}

/**Similar to foldin(), but the output has been preallocated.

     \ingroup Tensors
  */
void foldin_into(Tensor<double> &output, const Tensor<double> &a, int _ndx1,
                 const Tensor<double> &b, int _ndx2) {
  do_foldin_into(output, a, _ndx1, b, _ndx2);
}

}  // namespace tensor
