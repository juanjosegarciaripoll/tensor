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

#include <cassert>
#include <tensor/tensor.h>

namespace tensor {

template <typename n>
inline Tensor<n> do_adjoint(const Tensor<n> &a) {
  assert(a.rank() == 2);
  index rows = a.rows();
  index cols = a.columns();
  Tensor<n> b(cols, rows);
  if (cols && rows) {
    typename Tensor<n>::const_iterator ij_a = a.begin();
    typename Tensor<n>::iterator j_b = b.begin();
    for (index j = cols; j--; j_b++) {
      typename Tensor<n>::iterator ji_b = j_b;
      for (index i = rows; i--; ij_a++, ji_b += cols) {
        //assert(ij_a >= a.begin() && ij_a < a.end());
        //assert(ji_b >= b.begin() && ji_b < b.end());
        *ji_b = tensor::conj(*ij_a);
      }
    }
  }
  return b;
}

}  // namespace tensor
