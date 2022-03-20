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

#include <tensor/indices.h>

namespace tensor {

/** Concatenate two sets of indices by appending those of the second argument
   * to those of the first.
   *
   * Example:
   * \code
   * Indices i1{1,2};   // vector with elements (1,2)
   * Indices i2{3,4};   // vector with elements (3,4)
   * Indice itot = i1 << i2;        // vector with elements (1,2,3,4)
   * \endcode
   */
const Indices operator<<(const Indices &a, const Indices &b) {
  Indices output(a.size() + b.size());
  std::copy(a.cbegin(), a.cend(), output.begin());
  std::copy(b.cbegin(), b.cend(), output.begin() + a.size());
  return output;
}

}  // namespace tensor
