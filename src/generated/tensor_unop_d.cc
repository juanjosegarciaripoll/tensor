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

RTensor abs(const RTensor &t) {
  return ufunc1(t, [](double r) { return std::abs(r); });
}

RTensor sqrt(const RTensor &t) {
  return ufunc1(t, [](double r) { return std::sqrt(r); });
}

RTensor sin(const RTensor &t) {
  return ufunc1(t, [](double r) { return std::sin(r); });
}

RTensor cos(const RTensor &t) {
  return ufunc1(t, [](double r) { return std::cos(r); });
}

RTensor tan(const RTensor &t) {
  return ufunc1(t, [](double r) { return std::tan(r); });
}

RTensor exp(const RTensor &t) {
  return ufunc1(t, [](double r) { return std::exp(r); });
}

RTensor log(const RTensor &t) {
  return ufunc1(t, [](double r) { return std::log(r); });
}

RTensor sinh(const RTensor &t) {
  return ufunc1(t, [](double r) { return std::sinh(r); });
}

RTensor cosh(const RTensor &t) {
  return ufunc1(t, [](double r) { return std::cosh(r); });
}

RTensor tanh(const RTensor &t) {
  return ufunc1(t, [](double r) { return std::tanh(r); });
}

RTensor pow(const RTensor &t, double expt) {
  return ufunc1(t, [=](double r) { return std::pow(r, expt); });
}

}  // namespace tensor
