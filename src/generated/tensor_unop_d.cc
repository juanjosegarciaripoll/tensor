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

#include <cmath>
#include <tensor/tensor.h>
#include <tensor/tensor/mapping.h>

namespace tensor {

using tensor::mapping::ufunc1;

RTensor abs(const RTensor &t) { return ufunc1(t, ::fabs); }

RTensor sqrt(const RTensor &t) { return ufunc1(t, ::sqrt); }

RTensor sin(const RTensor &t) { return ufunc1(t, ::sin); }

RTensor cos(const RTensor &t) { return ufunc1(t, ::cos); }

RTensor tan(const RTensor &t) { return ufunc1(t, ::tan); }

RTensor exp(const RTensor &t) { return ufunc1(t, ::exp); }

RTensor log(const RTensor &t) { return ufunc1(t, ::log); }

RTensor sinh(const RTensor &t) { return ufunc1(t, ::sinh); }

RTensor cosh(const RTensor &t) { return ufunc1(t, ::cosh); }

RTensor tanh(const RTensor &t) { return ufunc1(t, ::tanh); }

RTensor pow(const RTensor &t, double expt) {
  return ufunc1(t, [=](double r) { return std::pow(r, expt); });
}

}  // namespace tensor
