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

#include <numeric>
#include <limits>
#include <functional>
#include <tensor/io.h>
#include <tensor/indices.h>

namespace tensor {

template class Vector<index>;

template class SimpleVector<index>;

bool all_equal(const Indices &a, const Indices &b) {
  return (a.size() == b.size()) && std::equal(a.cbegin(), a.cend(), b.cbegin());
}

index Dimensions::compute_total_size(const SimpleVector<index> &dims) {
  if (dims.size()) {
    index total_dimension = 1,
          maximum_dimension = std::numeric_limits<index>::max();
    for (index dimension : dims) {
      tensor_assert(dimension >= 0);
      tensor_assert(dimension < maximum_dimension);
      total_dimension *= dimension;
      if (dimension) {
        maximum_dimension /= dimension;
      }
    }
    return total_dimension;
  } else {
    return 0;
  }
}

Indices::Indices(const Dimensions &dims)
    : Indices(static_cast<size_t>(dims.rank())) {
  std::copy(dims.begin(), dims.end(), begin());
}

const Indices Indices::range(index min, index max, index step) {
  if (max < min) {
    return Indices();
  } else {
    auto size = static_cast<size_t>((max - min) / step + 1);
    Indices output(size);
    for (Indices::iterator it = output.begin(); it != output.end(); it++) {
      *it = min;
      min += step;
    }
    return output;
  }
}

}  // namespace tensor
