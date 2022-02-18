#pragma once
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

#ifndef TENSOR_IO_H
#define TENSOR_IO_H

#include <iostream>
#include <tensor/tensor.h>
#include <tensor/sparse.h>

namespace tensor {

/* Printing of complex numbers */
inline std::istream &operator>>(std::istream &s, cdouble &z) {
  double r, i;
  s >> r >> i;
  z = to_complex(r, i);
  return s;
}

inline std::ostream &operator<<(std::ostream &s, const cdouble &d) {
  return s << real(d) << ' ' << imag(d);
}

/**Simple text representation of vector.*/
template <typename elt_t>
inline std::ostream &operator<<(std::ostream &s, const Vector<elt_t> &t) {
  s << '[';
  const char *comma = "";
  for (auto &x : t) {
    s << comma << x;
    comma = ",";
  }
  return s << ']';
}

/**Simple text representation of vector.*/
template <typename elt_t>
inline std::ostream &operator<<(std::ostream &s, const SimpleVector<elt_t> &t) {
  s << '[';
  const char *comma = "";
  for (auto &x : t) {
    s << comma << x;
    comma = ",";
  }
  return s << ']';
}

/**Simple text representation of dimensions.*/
inline std::ostream &operator<<(std::ostream &s, const Dimensions &d) {
  return s << d.get_vector();
}

/**Simple text representation of tensor.*/
template <typename elt_t>
std::ostream &operator<<(std::ostream &s, const Tensor<elt_t> &t) {
  return s << '(' << t.dimensions() << ")/" << static_cast<Vector<elt_t>>(t);
}

template <typename t, size_t n>
inline std::ostream &operator<<(std::ostream &s, const StaticVector<t, n> &v) {
  return s << Vector<t>(v);
}

/**Simple text representation of a sparse matrix (undistinguishable from a tensor).*/
template <typename elt_t>
inline std::ostream &operator<<(std::ostream &s, const Sparse<elt_t> &t) {
  return s << full(t);
}

/* The following is a template used for creating an object that displays
 * a tensor with a slightly more attractive representation. We need to
 * create a template and then specializations because templates do not do
 * implicit type coercion. */
template <typename elt_t>
class MatrixForm {
  const Tensor<elt_t> data;
  MatrixForm();

 public:
  /**Creates an object that displays a tensor as a matrix.*/
  MatrixForm(const Tensor<elt_t> &t) : data(t) {}
  std::ostream &display(std::ostream &s) const;
};

template <typename elt_t>
inline std::ostream &operator<<(std::ostream &s, const MatrixForm<elt_t> &m) {
  return m.display(s);
}

/**Matrix form representation of a tensor.*/
const MatrixForm<double> matrix_form(const RTensor &t);

/**Matrix form representation of a tensor.*/
const MatrixForm<cdouble> matrix_form(const CTensor &t);

/**Matrix form representation of a tensor.*/
const MatrixForm<double> matrix_form(const RSparse &t);

/**Matrix form representation of a tensor.*/
const MatrixForm<cdouble> matrix_form(const CSparse &t);

/**Print a textual representation of a range.*/
std::ostream &operator<<(std::ostream &out, const Range &r);

/**Print a textual representation of a range iterator.*/
std::ostream &operator<<(std::ostream &out, const RangeIterator &it);

}  // namespace tensor

#endif  // !TENSOR_IO_H
