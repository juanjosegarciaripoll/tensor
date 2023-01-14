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
#include <tensor/tensor.h>

namespace tensor {

/**Complex conjugate of a tensor.*/
const CTensor conj(const CTensor &t) {
  auto output = CTensor::empty(t.dimensions());
  std::transform(t.cbegin(), t.cend(), output.unsafe_begin_not_shared(),
				 static_cast<cdouble(*)(cdouble)>(tensor::conj));
  return output;
}

}  // namespace tensor
