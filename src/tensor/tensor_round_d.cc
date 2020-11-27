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
#include <functional>
#include <algorithm>
#include <tensor/tensor.h>

namespace tensor {

#if defined(_MSC_VER) && (_MSC_VER < 1800)
double tensor::round(double x) {
  return floor((x < 0) ? (x - 0.5) : (x + 0.5));
}
#endif

// Creation of a user-defined function object
// that inherits from the unary_function base class
class round1 : std::unary_function<double, double> {
 public:
  result_type operator()(argument_type i) {
#if defined(_MSC_VER) && (_MSC_VER < 1800)
    return round(i);
#else
    return ::round(i);
#endif
  }
};

RTensor round(const RTensor &r) {
  RTensor output(r.dimensions());
  std::transform(r.begin(), r.end(), output.begin(), round1());
  return output;
}

}  // namespace tensor
