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

namespace tensor {

Tensor<cdouble> fold(const Tensor<double> &a, int ndx1,
                     const Tensor<cdouble> &b, int ndx2) {
  return fold(to_complex(a), ndx1, b, ndx2);
}

const Tensor<cdouble> foldc(const Tensor<double> &a, int ndx1,
                            const Tensor<cdouble> &b, int ndx2) {
  return fold(to_complex(a), ndx1, b, ndx2);
}

Tensor<cdouble> mmult(const Tensor<double> &m1, const Tensor<cdouble> &m2) {
  return fold(to_complex(m1), -1, m2, 0);
}

Tensor<cdouble> fold(const Tensor<cdouble> &a, int ndx1,
                     const Tensor<double> &b, int ndx2) {
  return fold(a, ndx1, to_complex(b), ndx2);
}

const Tensor<cdouble> foldc(const Tensor<cdouble> &a, int ndx1,
                            const Tensor<double> &b, int ndx2) {
  return foldc(a, ndx1, to_complex(b), ndx2);
}

Tensor<cdouble> mmult(const Tensor<cdouble> &m1, const Tensor<double> &m2) {
  return fold(m1, -1, to_complex(m2), 0);
}

}  // namespace tensor
