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

  double matrix_norminf(const RTensor &m)
  {
    assert(m.rank() == 2);

    size_t r = m.rows();
    size_t c = m.columns();

    // aux[i] = sum_j abs(A(i,j))

    double *aux = new double[r];
    RTensor::const_iterator p = m.begin_const();
    for (size_t i = 0; i < r; i++, p++) {
      aux[i] = abs(*p);
    }
    for (size_t j = 1; j < c; j++) {
      for (size_t i = 0; i < r; i++, p++) {
	aux[i] += abs(*p);
      }
    }

    // output = max_i aux[i]

    double output = 0.0;
    for (size_t i = 0; i < r; i++) {
      if (output < aux[i])
	output = aux[i];
    }
    delete[] aux;

    return output;
  }

} // namespace tensor
