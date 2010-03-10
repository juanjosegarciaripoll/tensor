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

#include <numeric>
#include <functional>
#include <tensor/indices.h>

namespace tensor {

  const ListGenerator<index> igen = {};
  const ListGenerator<double> rgen = {};
  const ListGenerator<cdouble> cgen = {};

  template class Vector<index>;

  bool Indices::operator==(const Indices &other) const {
    return std::equal(other.begin_const(), other.end_const(), this->begin_const());
  }

  index Indices::total_size() const {
    if (size()) {
      return std::accumulate(begin_const(), end_const(),
                             static_cast<index>(1), std::multiplies<index>());
    } else {
      return 0;
    }
  }


} // namespace
