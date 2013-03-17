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
#include <mps/rmps.h>
#include "mps_presize.h"

namespace mps {

  template class MP<tensor::RTensor>;

  bool RMPS::is_periodic() const {
    index l = size();
    if (l) {
      index d0 = (*this)[0].dimension(0);
      index dl = (*this)[l-1].dimension(2);
      if (d0 == dl && d0 > 1)
	return true;
    }
    return false;
  }

  index RMPS::normal_index(index mps_index) const {
    index mps_size = size();
    if (mps_index < 0) {
      assert(mps_index >= -mps_size);
      return mps_index + mps_size;
    } else {
      assert(mps_index < mps_size);
      return mps_index;
    }
  }

  RMPS::RMPS() :
    parent()
  {
  }

  RMPS::RMPS(index length, index physical_dimension, index bond_dimension,
	     bool periodic) :
    parent(length)
  {
    if (physical_dimension) {
      tensor::Indices d(length);
      std::fill(d.begin(), d.end(), physical_dimension);
      presize_mps(*this, d, bond_dimension, periodic);
    }
  }

  RMPS::RMPS(const tensor::Indices &physical_dimensions, index bond_dimension,
	     bool periodic) :
    parent(physical_dimensions.size())
  {
    presize_mps(*this, physical_dimensions, bond_dimension, periodic);
  }

} // namespace mps
