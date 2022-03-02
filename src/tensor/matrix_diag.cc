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

#include <tensor/exceptions.h>
#include <tensor/tensor.h>

namespace tensor {

template <typename n>
inline Tensor<n> do_diag(const Tensor<n> &a, int which, index rows,
                         index cols) {
  auto output = Tensor<n>::empty(rows, cols);
  output.fill_with_zeros();
  index r0, c0;
  if (which < 0) {
    r0 = -which;
    c0 = 0;
  } else {
    r0 = 0;
    c0 = which;
  }
  index l = std::min<index>(rows - r0, cols - c0);
  tensor_assert2(
      l >= 0, std::invalid_argument(
                  "In diag(a,which,...) the value of WHICH exceeds the size of "
                  "the matrix"));
  tensor_assert2(l == a.ssize(),
                 std::invalid_argument(
                     "In diag(A,...) the vector a has too few/many elements."));
  for (index i = 0; i < l; i++) {
    output.at(r0 + i, c0 + i) = a[i];
  }
  return output;
}

}  // namespace tensor
