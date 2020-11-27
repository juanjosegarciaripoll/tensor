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

template <class Tensor>
static const Tensor change_dimension_inner(const Tensor &a, int dim,
                                           index new_size) {
  typedef typename Tensor::elt_t elt_t;

  dim = normalize_index(dim, a.rank());
  Indices d = a.dimensions();
  index old_size = d[dim];
  if (old_size == new_size) return a;

  d.at(dim) = new_size;
  Tensor output(d);
  if (new_size > a.dimension(dim)) output.fill_with_zeros();

  typename Tensor::iterator p_new = output.begin();
  typename Tensor::const_iterator p_old = a.begin();
  index i_len, k_len;
  surrounding_dimensions(d, dim, &i_len, &new_size, &k_len);

  index dp_new = new_size * i_len;
  index dp_old = old_size * i_len;
  index data_size = std::min(dp_new, dp_old) * sizeof(*p_new);
  while (k_len--) {
    memcpy(p_new, p_old, data_size);
    p_old += dp_old;
    p_new += dp_new;
  }
  return output;
}

}  // namespace tensor
