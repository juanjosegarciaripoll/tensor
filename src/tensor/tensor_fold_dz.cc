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

namespace tensor {

CTensor fold(const RTensor &a, int ndx1, const CTensor &b, int ndx2) {
  return fold(to_complex(a), ndx1, b, ndx2);
}

CTensor foldc(const RTensor &a, int ndx1, const CTensor &b, int ndx2) {
  return fold(to_complex(a), ndx1, b, ndx2);
}

CTensor mmult(const RTensor &m1, const CTensor &m2) {
  return fold(to_complex(m1), -1, m2, 0);
}

CTensor fold(const CTensor &a, int ndx1, const RTensor &b, int ndx2) {
  return fold(a, ndx1, to_complex(b), ndx2);
}

CTensor foldc(const CTensor &a, int ndx1, const RTensor &b, int ndx2) {
  return foldc(a, ndx1, to_complex(b), ndx2);
}

CTensor mmult(const CTensor &m1, const RTensor &m2) {
  return fold(m1, -1, to_complex(m2), 0);
}

}  // namespace tensor
