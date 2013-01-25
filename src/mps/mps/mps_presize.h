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

#include <cassert>
#include <tensor/tensor.h>

template<class MP>
static void
presize_mps(MP &mp, const tensor::Indices &physical_dimensions,
	    tensor::index bond_dimension, bool periodic)
{
  assert(bond_dimension > 0);
  tensor::index l = physical_dimensions.size();
  tensor::Indices dimensions = tensor::igen << bond_dimension << 0 << bond_dimension;
  for (tensor::index i = 0; i < l; i++) {
    dimensions.at(1) = physical_dimensions[l];
    if (!periodic || (i > 0) || (i < (l-1))) {
      dimensions.at(0) = dimensions.at(2) = bond_dimension;
    } else if (i == 0) {
      dimensions.at(0) = 1;
      dimensions.at(2) = bond_dimension;
    } else {
      dimensions.at(0) = bond_dimension;
      dimensions.at(2) = 1;
    }
    mp.at(i) = MP::elt_t::random(dimensions);
  }
}
