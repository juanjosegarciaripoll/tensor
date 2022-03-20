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

#include <stdexcept>
#include <tensor/exceptions.h>
#include <tensor/indices.h>
#include <tensor/io.h>

namespace tensor {

std::ostream &operator<<(std::ostream &out, const Range &r) {
  out << "Range(dim=" << r.dimension() << ',';
  if (!r.has_indices()) {
    out << "start=" << r.first() << ",last=" << r.last() << ",step=" << r.step()
        << ",dim=" << r.dimension();
  } else {
    out << "indices=" << r.indices();
  }
  return out << ",size=" << r.size() << ')';
}

std::ostream &operator<<(std::ostream &out, const RangeIterator &r) {
  out << "RangeIterator(counter=" << r.counter() << ",offset=" << r.offset()
      << ",limit=" << r.limit() << ",step=" << r.step();
  if (r.has_indices()) {
    out << ",indices=" << r.indices();
  }
  out << ",next=";
  if (r.has_next()) {
    out << r.next();
  } else {
    out << "nullptr";
  }
  return out << ')';
}

}  // namespace tensor
