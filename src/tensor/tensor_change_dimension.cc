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
#include <tensor/detail/common.h>

namespace tensor {

  template<class Tensor>
  static const Tensor change_dimension_inner(const Tensor &a, int dim, index new_size)
  {
    typedef typename Tensor::elt_t elt_t;

    dim = normalize_index(dim, a.rank());
    Indices d = a.dimensions();
    index old_size = d[dim];
    if (old_size == new_size)
      return a;

    d.at(dim) = new_size;
    Tensor output(d);
    elt_t *p1 = output.begin();
    const elt_t *p2 = a.begin();
    index i_len, k_len;
    surrounding_dimensions(d, dim, &i_len, &new_size, &k_len);
    if (old_size > new_size) {
      output.fill_with_zeros();
      index dp2 = (old_size - new_size) * i_len;
      for (index k = 0; k < k_len; k++) {
	for (index j = 0; j < new_size; j++) {
	  memcpy(p1, p2, i_len * sizeof(*p1));
	  p1 += i_len;
	  p2 += i_len;
	}
	p2 += dp2;
      }
    } else {
      index dp1 = (new_size - old_size) * i_len;
      for (index k = 0; k < k_len; k++) {
	for (index j = 0; j < old_size; j++) {
	  memcpy(p1, p2, i_len * sizeof(*p1));
	  p1 += i_len;
	  p2 += i_len;
	}
	for (index i = 0; i < dp1; i++, p1++)
	  *p1 = number_zero<elt_t>();
      }
    }
    return output;
  }

} // namespace tensor
