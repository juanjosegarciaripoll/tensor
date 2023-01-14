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

CTensor to_complex(const RTensor &r) {
  auto output = CTensor::empty(r.dimensions());
  std::transform(r.cbegin(), r.cend(), output.unsafe_begin_not_shared(),
                 [](double x) { return to_complex(x); });
  return output;
}

CTensor to_complex(const RTensor &r, const RTensor &i) {
  // This should be: tensor_assert(r.dimensions() == i.dimensions())
  tensor_assert(r.size() == i.size());
  auto output = CTensor::empty(r.dimensions());
  std::transform(r.cbegin(), r.cend(), i.cbegin(),
                 output.unsafe_begin_not_shared(),
                 [](double x, double y) { return to_complex(x, y); });
  return output;
}

}  // namespace tensor
