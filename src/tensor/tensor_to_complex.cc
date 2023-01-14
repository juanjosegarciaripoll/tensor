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

#define TENSOR_LOAD_IMPL
#include <tensor/tensor.h>

namespace tensor {

// FIXME use algorithms

CTensor to_complex(const RTensor &r) {
  auto output = CTensor::empty(r.dimensions());
  RTensor::const_iterator ir = r.begin();
  for (CTensor::iterator io = output.unsafe_begin_not_shared();
       io != output.unsafe_end_not_shared(); ++io, ++ir) {
    *io = to_complex(*ir);
  }
  return output;
}

CTensor to_complex(const RTensor &r, const RTensor &i) {
  // This should be: tensor_assert(verify_tensor_dimensions_match(a.dimensions(), b.dimensions()));
  tensor_assert(r.size() == i.size());
  auto output = CTensor::empty(r.dimensions());
  RTensor::const_iterator ir = r.begin();
  RTensor::const_iterator ii = i.begin();
  for (CTensor::iterator io = output.unsafe_begin_not_shared();
       io != output.unsafe_end_not_shared(); ++io, ++ir, ++ii) {
    *io = to_complex(*ir, *ii);
  }
  return output;
}

}  // namespace tensor
