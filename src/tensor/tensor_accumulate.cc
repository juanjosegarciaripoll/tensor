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

#include <tensor/tensor/accumulate.h>

namespace tensor {

AccumulateDimensions prepare_accumulate(const Dimensions &dimensions,
                                        index axis) {
  axis = Dimensions::normalize_index(axis, dimensions.rank());
  tensor_assert(axis < dimensions.rank() && axis >= 0);
  SimpleVector<index> output_dimensions(dimensions.rank() - 1);
  index left_size = 1;
  for (index i = 0; i < axis; ++i) {
    auto n = output_dimensions[i] = dimensions[i];
    left_size *= n;
  }
  index size = dimensions[axis];
  index right_size = 1;
  for (index i = axis + 1; i < dimensions.rank(); ++i) {
    auto n = output_dimensions[i - 1] = dimensions[i];
    right_size *= n;
  }
  if (output_dimensions.size() == 0) {
    return AccumulateDimensions{Dimensions{1}, left_size, size, right_size};
  } else {
    return AccumulateDimensions{output_dimensions, left_size, size, right_size};
  }
}

}  // namespace tensor
