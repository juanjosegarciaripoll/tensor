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
#include "tensor_kron.hpp"

namespace tensor {

/*!This products builds, out of matrices \c A
   and \c B of dimensions \c MxM and \c NxN, a bigger matrix of size \c (MxN)^2.
   Roughly, the formula for such a matrix is
   \f[
   C_{i+Mj,k+Ml} = A_{ik} B_{jl} =: A \otimes B.
   \f]

   \ingroup Linalg
   For example, if A = [1,0;0,2] and B = [1,2;3,4] then the kronecker product
   of both matrices is \verbatim
	C = [1, 2, 0, 0;
	     3, 4, 0, 0;
	     0, 0, 2, 4;
	     0, 0, 6, 8];
   \endverbatim
*/
RTensor kron(const RTensor &s1, const RTensor &s2) { return do_kron(s1, s2); }

RTensor kron2(const RTensor &s1, const RTensor &s2) { return do_kron(s2, s1); }

}  // namespace tensor
