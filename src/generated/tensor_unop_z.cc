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
  return ufunc1(t, static_cast<double(*)(const std::complex<double> &z)>(std::abs));
}

CTensor sqrt(const CTensor &t) {
  return ufunc1(t, static_cast<cdouble(*)(const cdouble &z)>(std::sqrt));
}

CTensor sin(const CTensor &t) {
  return ufunc1(t, static_cast<cdouble(*)(const cdouble &z)>(std::sin));
}

CTensor cos(const CTensor &t) {
  return ufunc1(t, static_cast<cdouble(*)(const cdouble &z)>(std::cos));
}

CTensor tan(const CTensor &t) {
  return ufunc1(t, static_cast<cdouble(*)(const cdouble &z)>(std::tan));
}

CTensor exp(const CTensor &t) {
  return ufunc1(t, static_cast<cdouble(*)(const cdouble &z)>(std::exp));
}

CTensor log(const CTensor &t) {
  return ufunc1(t, static_cast<cdouble(*)(const cdouble &z)>(std::log));
}

CTensor sinh(const CTensor &t) {
  return ufunc1(t, static_cast<cdouble(*)(const cdouble &z)>(std::sinh));
}

CTensor cosh(const CTensor &t) {
  return ufunc1(t, static_cast<cdouble(*)(const cdouble &z)>(std::cosh));
}

CTensor tanh(const CTensor &t) {
  return ufunc1(t, static_cast<cdouble(*)(const cdouble &z)>(std::tanh));
}

CTensor pow(const CTensor &t, double expt) {
  return ufunc1(t, [=](cdouble r) { return std::pow(r, expt); });
}

CTensor pow(const CTensor &t, cdouble expt) {
  return ufunc1(t, [=](cdouble r) { return std::pow(r, expt); });
}

}  // namespace tensor
