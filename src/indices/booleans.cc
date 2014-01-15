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

#include <tensor/indices.h>

namespace tensor {

  template class Vector<bool>;

  class Booleans;

  const ListGenerator<bool> bgen = {};

  bool all_of(const Booleans& b)
  {
    return std::count(b.begin_const(), b.end_const(), true) == b.size();
  }

  bool any_of(const Booleans& b)
  {
    return std::find(b.begin_const(), b.end_const(), true) != b.end_const();
  }

} // namespace tensor
