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

#ifndef TENSOR_LINSPACE_HPP
#define TENSOR_LINSPACE_HPP

#include <tensor/gen.h>

namespace tensor {

template <typename elt_t>
static inline const Tensor<elt_t> do_linspace(const Tensor<elt_t> &min,
                                              const Tensor<elt_t> &max,
                                              index n = 100) {
  index d = min.ssize();
  auto output = Tensor<elt_t>::empty(d, n);
  if (n == 1) {
    output = min;
  } else if (n) {
    const Tensor<elt_t> base = reshape(max, d);
    const Tensor<elt_t> delta =
        reshape((max - min) / static_cast<double>(n - 1), d);
    for (index i = 0; i < n; i++) {
      output.at(_, range(i)) = delta * static_cast<double>(i) + min;
    }
  }
  if (d == 1)
    return reshape(output, n);
  else
    return reshape(output, min.dimensions() << (igen << n));
}

}  // namespace tensor

#endif  // TENSOR_LINSPACE_HPP
