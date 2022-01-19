// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
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

#include <tensor/tensor.h>

namespace tensor {

RTensor operator-(const RTensor &a, double b) {
  RTensor output(a.dimensions());
  RTensor::const_iterator ita = a.begin();
  RTensor::iterator dest = output.begin();
  for (index i = a.size(); i; --i, ++dest, ++ita) {
    *dest = (*ita) - (b);
  }
  return output;
}

RTensor operator-(double a, const RTensor &b) {
  RTensor output(b.dimensions());
  RTensor::const_iterator itb = b.begin();
  RTensor::iterator dest = output.begin();
  for (index i = b.size(); i; --i, ++dest, ++itb) {
    *dest = (a) - (*itb);
  }
  return output;
}

}  // namespace tensor
