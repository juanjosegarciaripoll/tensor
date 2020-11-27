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

#include <tensor/indices.h>

namespace tensor {

/** Calculates the number of elements before, at and after a given index.
   *
   * @param d    a dimension vector that is studied (usually, got by Tensor::dimensions())
   * @param ndx  the index of an element of d.
   * @param d1   the value of the pointer position is set to the product of all
   *             d[i] with \f$ i < ndx \f$.
   * @param d2   the value at this pointer position is set to i[ndx].
   * @param d3   the value at this pointer position is set to the product of all
   *             d[i] with \f$ i > ndx \f$
   */
void surrounding_dimensions(const Indices &d, index ndx, index *d1, index *d2,
                            index *d3) {
  index i, l = d.size();
  for (i = 0, *d1 = 1; i < ndx; i++) {
    *d1 *= d[i];
  }
  *d2 = d[i++];
  for (*d3 = 1; i < l; i++) {
    *d3 *= d[i];
  }
}

}  // namespace tensor
