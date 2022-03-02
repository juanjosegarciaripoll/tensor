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

template <typename n>
static const n do_trace(const Tensor<n> &matrix) {
  tensor_assert(matrix.rank() == 2);
  n output = number_zero<n>();
  const index r = matrix.rows();
  const index c = matrix.columns();
  typename Tensor<n>::const_iterator it = matrix.begin();
  for (index j = std::min(r, c); j--; it += (r + 1)) {
    output += *it;
  }
  return output;
}

}  // namespace tensor
