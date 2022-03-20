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

#include <math.h>
#include <cmath>
#include <algorithm>
#include <tensor/tensor.h>

namespace tensor {

RTensor round(const RTensor &t) {
  RTensor output(t.dimensions());
#if defined(_MSC_VER) && (_MSC_VER < 1800)
  std::transform(t.begin(), t.end(), output.begin(), [](double x) {
    return floor((x < 0) ? (x - 0.5) : (x + 0.5));
  });
#else
  std::transform(t.begin(), t.end(), output.begin(),
                 [](double x) { return ::round(x); });
#endif
  return output;
}

}  // namespace tensor
