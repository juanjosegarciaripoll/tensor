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

const TYPE3 OPERATOR1(const TYPE1 &a, const TYPE2 &b) {
  assert(a.size() == b.size());
  TYPE3 output(a.dimensions());
  TYPE1::const_iterator ita = a.begin();
  TYPE2::const_iterator itb = b.begin();
  TYPE3::iterator dest = output.begin();
  for (index i = a.size(); i; --i, ++dest, ++ita, ++itb) {
    *dest = (*ita)OPERATOR2(*itb);
  }
  return output;
}

}  // namespace tensor
