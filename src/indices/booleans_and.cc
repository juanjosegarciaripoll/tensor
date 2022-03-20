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

#include <functional>
#include <algorithm>
#include <tensor/indices.h>
#include <tensor/exceptions.h>

namespace tensor {

Booleans operator&&(const Booleans &a, const Booleans &b) {
  tensor_assert(a.size() == b.size());
  Booleans output(a.size());
  std::transform(a.begin(), a.end(), b.begin(), output.begin(),
                 std::logical_and<>());

  return output;
}

}  // namespace tensor
