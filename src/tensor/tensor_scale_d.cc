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

#include "tensor_scale.cc"

namespace tensor {

/**Hadamard product of a tensor times a vector. A Hadamard product
     consists on multiplying the element that have equal indices. For
     instance, if the tensor T has three indices, the call \c
     {scale(t, 1, v)} would be equivalent to \f$t_{i,j,k}v_j\f$.

     \ingroup Tensors
  */
const Tensor<double> scale(const Tensor<double> &t, int ndx,
                           const Tensor<double> &v) {
  index d1, d2, d3;
  Tensor<double> output(t.dimensions());
  ndx = normalize_index(ndx, t.rank());
  surrounding_dimensions(t.dimensions(), ndx, &d1, &d2, &d3);
  if (d2 != v.size()) {
    std::cerr << "In scale() the dimension " << ndx
              << " of the tensor does not match the length " << v.size()
              << " of the scale vector" << std::endl;
    abort();
  }
  doscale(output.begin(), t.begin_const(), v.begin_const(), d1, d2, d3);
  return output;
}

void scale_inplace(Tensor<double> &t, int ndx, const Tensor<double> &v) {
  index d1, d2, d3;
  surrounding_dimensions(t.dimensions(), normalize_index(ndx, t.rank()), &d1,
                         &d2, &d3);
  if (d2 != v.size()) {
    std::cerr << "In scale() the dimension " << ndx
              << " of the tensor does not match the length " << v.size()
              << " of the scale vector" << std::endl;
    abort();
  }
  doscale(t.begin(), v.begin_const(), d1, d2, d3);
}

}  // namespace tensor
