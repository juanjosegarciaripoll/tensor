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

#include <algorithm>
#include <functional>
#include <tensor/tensor.h>

namespace tensor {

/** \todo Evaluate whether index comparisons may be turned into Vector
	comparisons and serve as basis for tensor comparisons as well. */
template <typename t1, typename t2, class comparison>
static Booleans compare_vectors(const Vector<t1> &a, const Vector<t2> &b,
                                comparison fn) {
  tensor_assert(a.size() == b.size());
  Booleans output(a.size());
  std::transform(a.cbegin(), a.cend(), b.cbegin(), output.begin(), fn);
  return output;
}

template <typename t1, typename t2, class comparison>
static Booleans compare_vector_and_number(const Vector<t1> &a, t2 b,
                                          comparison fn) {
  Booleans output(a.size());
  std::transform(a.cbegin(), a.cend(), output.begin(),
                 [&](const auto &value) { return fn(value, b); });
  return output;
}

Booleans operator<(const Indices &a, const Indices &b) {
  return compare_vectors(a, b, std::less<>());
}

Booleans operator>(const Indices &a, const Indices &b) {
  return compare_vectors(a, b, std::greater<>());
}

Booleans operator<=(const Indices &a, const Indices &b) {
  return compare_vectors(a, b, std::less_equal<>());
}

Booleans operator>=(const Indices &a, const Indices &b) {
  return compare_vectors(a, b, std::greater_equal<>());
}

Booleans operator==(const Indices &a, const Indices &b) {
  return compare_vectors(a, b, std::equal_to<>());
}

Booleans operator!=(const Indices &a, const Indices &b) {
  return compare_vectors(a, b, std::not_equal_to<>());
}

//
// VECTOR-NUMBER
//

Booleans operator<(const Indices &a, index b) {
  return compare_vector_and_number(a, b, std::less<>());
}

Booleans operator>(const Indices &a, index b) {
  return compare_vector_and_number(a, b, std::greater<>());
}

Booleans operator<=(const Indices &a, index b) {
  return compare_vector_and_number(a, b, std::less_equal<>());
}

Booleans operator>=(const Indices &a, index b) {
  return compare_vector_and_number(a, b, std::greater_equal<>());
}

Booleans operator==(const Indices &a, index b) {
  return compare_vector_and_number(a, b, std::equal_to<>());
}

Booleans operator!=(const Indices &a, index b) {
  return compare_vector_and_number(a, b, std::not_equal_to<>());
}

bool all_equal(const Indices &a, const Indices &b) {
  return (a.size() == b.size()) && std::equal(a.cbegin(), a.cend(), b.cbegin());
}

bool all_equal(const Indices &a, index b) {
  return std::all_of(a.cbegin(), a.cend(),
                     [&](const auto &value) { return value == b; });
}

}  // namespace tensor
