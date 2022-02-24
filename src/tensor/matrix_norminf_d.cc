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

double matrix_norminf(const RTensor &m) {
  assert(m.rank() == 2);

  auto r = m.rows();
  auto c = m.columns();

  // aux[i] = sum_j abs(A(i,j))

  SimpleVector<double> aux(static_cast<size_t>(r));
  auto p = m.cbegin();
  for (index i = 0; i < r; i++, ++p) {
    aux.at(i) = std::abs(*p);
  }
  for (index j = 1; j < c; j++) {
    for (index i = 0; i < r; i++, ++p) {
      aux.at(i) += std::abs(*p);
    }
  }
  return *std::max_element(std::begin(aux), std::end(aux));
}

}  // namespace tensor
