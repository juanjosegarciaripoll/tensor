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

#if !defined(TENSOR_IO_H) || defined(TENSOR_DETAIL_IO_HPP)
#error "This header cannot be included manually"
#else
#define TENSOR_DETAIL_IO_HPP

namespace tensor {

template <class ForwardIterator>
void write_to_stream(std::ostream &s, ForwardIterator begin,
                     ForwardIterator end) {
  bool first = true;
  for (bool first = true; begin != end; ++begin, first = false) {
    if (!first) s << ", ";
    s << *begin;
  }
}

template <typename elt_t>
std::ostream &operator<<(std::ostream &s, const Vector<elt_t> &t) {
  s << "[";
  write_to_stream(s, t.begin_const(), t.end_const());
  s << "]";
  return s;
}

template <typename elt_t>
std::ostream &operator<<(std::ostream &s, const Tensor<elt_t> &t) {
  s << "(" << t.dimensions() << ")/[";
  write_to_stream(s, t.begin_const(), t.end_const());
  s << "]";
  return s;
}

}  // namespace tensor

#endif  // !TENSOR_DETAIL_IO_HPP
