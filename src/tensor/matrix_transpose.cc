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
#include "tensor_permute.cc"

namespace tensor {

template <typename n>
inline Tensor<n> do_transpose(const Tensor<n> &a) {
  tensor_assert(a.rank() == 2);
  index rows = a.rows();
  index cols = a.columns();
  auto b = Tensor<n>::empty(cols, rows);
  if (cols && rows) permute_12(b, a, rows, cols, 1);
  return b;
}

}  // namespace tensor
