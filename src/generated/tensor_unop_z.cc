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
#include <complex>
#include <tensor/tensor.h>
#include <tensor/tensor/mapping.h>

namespace tensor {

using tensor::mapping::ufunc1;

RTensor abs(const CTensor &t) {
  auto output = RTensor::empty(t.dimensions());
  std::transform(t.cbegin(), t.cend(), output.begin(),
                 [](const cdouble &z) -> double { return std::abs(z); });
  return output;
}

CTensor sqrt(const CTensor &t) {
  return ufunc1(t, [](const cdouble &z) { return std::sqrt(z); });
}

CTensor sin(const CTensor &t) {
  return ufunc1(t, [](const cdouble &z) { return std::sin(z); });
}

CTensor cos(const CTensor &t) {
  return ufunc1(t, [](const cdouble &z) { return std::cos(z); });
}

CTensor tan(const CTensor &t) {
  return ufunc1(t, [](const cdouble &z) { return std::tan(z); });
}

CTensor exp(const CTensor &t) {
  return ufunc1(t, [](const cdouble &z) { return std::exp(z); });
}

CTensor log(const CTensor &t) {
  return ufunc1(t, [](const cdouble &z) { return std::log(z); });
}

CTensor sinh(const CTensor &t) {
  return ufunc1(t, [](const cdouble &z) { return std::sinh(z); });
}

CTensor cosh(const CTensor &t) {
  return ufunc1(t, [](const cdouble &z) { return std::cosh(z); });
}

CTensor tanh(const CTensor &t) {
  return ufunc1(t, [](const cdouble &z) { return std::tanh(z); });
}

CTensor pow(const CTensor &t, double expt) {
  return ufunc1(t, [=](cdouble r) { return std::pow(r, expt); });
}

CTensor pow(const CTensor &t, cdouble expt) {
  return ufunc1(t, [=](cdouble r) { return std::pow(r, expt); });
}

}  // namespace tensor
