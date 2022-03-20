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

#include <algorithm>
#include <functional>
#include <tensor/tensor.h>

namespace tensor {

RTensor sort(const RTensor &v, bool reverse) {
  RTensor output(v);
  if (reverse) {
    std::sort(output.begin(), output.end(), std::greater<>());
  } else {
    std::sort(output.begin(), output.end(), std::less<>());
  }
  return output;
}

Indices sort_indices(const RTensor &v, bool reverse) {
  if (v.size()) {
    Indices output = iota(0, v.ssize() - 1);
    if (reverse) {
      std::sort(output.begin(), output.end(),
                [&](index i1, index i2) { return v[i1] > v[i2]; });
    } else {
      std::sort(output.begin(), output.end(),
                [&](index i1, index i2) { return v[i1] < v[i2]; });
    }
    return output;
  } else {
    return {};
  }
}

}  // namespace tensor
