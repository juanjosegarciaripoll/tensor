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
#include <tensor/tensor_blas.h>

namespace tensor {

#if 1

double norm2(const RTensor &r) {
  auto N = blas::size_t_to_blas(r.size());
  return cblas_dnrm2(N, blas::tensor_pointer(r), 1);
}

double scprod(const RTensor &a, const RTensor &b) {
  auto N = blas::size_t_to_blas(a.size());
  return cblas_ddot(N, blas::tensor_pointer(a), 1, blas::tensor_pointer(b), 1);
}

#else

double norm2(const RTensor &r) { return ::sqrt(scprod(r, r)); }

double scprod(const RTensor &a, const RTensor &b) {
  double output = 0;
  for (RTensor::const_iterator ia = a.begin(), ib = b.begin(); ia != a.end();
       ++ia, ++ib)
    output += (*ia) * (*ib);
  return output;
}

#endif

}  // namespace tensor
