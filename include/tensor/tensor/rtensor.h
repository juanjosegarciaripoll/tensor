// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
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
#ifndef TENSOR_TENSOR_RTENSOR_H
#define TENSOR_TENSOR_RTENSOR_H

#include <tensor/tensor/types.h>
#include <tensor/tensor/operators.h>

/*!\addtogroup Tensors*/
/* @{ */

namespace tensor {

extern template class Tensor<double>;
/** Real Tensor with elements of type "double". */
#ifdef DOXYGEN_ONLY
struct RTensor : public Tensor<double> {
}
#else
typedef Tensor<double> RTensor;
#endif

RTensor
change_dimension(const RTensor &U, int dimension, index new_size);

/**Return the smallest element in the tensor.*/
double min(const RTensor &r);
/**Return the largest element in the tensor.*/
double max(const RTensor &r);
/**Return the sum of the elements in the tensor.*/
double sum(const RTensor &r);
/**Return the mean of the elements in the tensor.*/
double mean(const RTensor &r);
/**Return the mean of the elements in the along the given dimension.*/
RTensor mean(const RTensor &r, int which);

double norm0(const RTensor &r);
double scprod(const RTensor &a, const RTensor &b);
double norm2(const RTensor &r);
double matrix_norminf(const RTensor &r);

RTensor abs(const RTensor &t);
RTensor cos(const RTensor &t);
RTensor sin(const RTensor &t);
RTensor tan(const RTensor &t);
RTensor cosh(const RTensor &t);
RTensor sinh(const RTensor &t);
RTensor tanh(const RTensor &t);
RTensor exp(const RTensor &t);
RTensor sqrt(const RTensor &t);
RTensor log(const RTensor &t);

RTensor round(const RTensor &t);

RTensor diag(const RTensor &d, int which, index rows, index cols);
RTensor diag(const RTensor &d, int which = 0);
RTensor take_diag(const RTensor &d, int which = 0, int ndx1 = 0, int ndx2 = -1);
double trace(const RTensor &d);
RTensor trace(const RTensor &A, int i1, int i2);

RTensor squeeze(const RTensor &t);
RTensor permute(const RTensor &a, index ndx1 = 0, index ndx2 = -1);
RTensor transpose(const RTensor &a);
inline RTensor adjoint(const RTensor &a) { return transpose(a); }

RTensor fold(const RTensor &a, int ndx1, const RTensor &b, int ndx2);
RTensor foldc(const RTensor &a, int ndx1, const RTensor &b, int ndx2);
RTensor foldin(const RTensor &a, int ndx1, const RTensor &b, int ndx2);
RTensor mmult(const RTensor &a, const RTensor &b);

RTensor scale(const RTensor &t, int ndx1, const RTensor &v);
void scale_inplace(RTensor &t, int ndx1, const RTensor &v);

void fold_into(RTensor &output, const RTensor &a, int ndx1, const RTensor &b,
               int ndx2);
void foldin_into(RTensor &output, const RTensor &a, int ndx1, const RTensor &b,
                 int ndx2);
void mmult_into(RTensor &output, const RTensor &a, const RTensor &b);

RTensor linspace(double min, double max, index n = 100);
RTensor linspace(const RTensor &min, const RTensor &max, index n = 100);

Indices sort(const Indices &v, bool reverse = false);
Indices sort_indices(const Indices &v, bool reverse = false);

RTensor sort(const RTensor &v, bool reverse = false);
Indices sort_indices(const RTensor &v, bool reverse = false);

bool all_equal(const RTensor &a, const RTensor &b);
bool all_equal(const RTensor &a, double b);
inline bool all_equal(double b, const RTensor &a) { return all_equal(a, b); }
template <typename t1, typename t2>
inline bool some_unequal(const t1 &a, const t2 &b) {
  return !all_equal(a, b);
}

Booleans operator==(const RTensor &a, const RTensor &b);
Booleans operator<(const RTensor &a, const RTensor &b);
Booleans operator>(const RTensor &a, const RTensor &b);
Booleans operator<=(const RTensor &a, const RTensor &b);
Booleans operator>=(const RTensor &a, const RTensor &b);
Booleans operator!=(const RTensor &a, const RTensor &b);

Booleans operator==(const RTensor &a, double b);
Booleans operator<(const RTensor &a, double b);
Booleans operator>(const RTensor &a, double b);
Booleans operator<=(const RTensor &a, double b);
Booleans operator>=(const RTensor &a, double b);
Booleans operator!=(const RTensor &a, double b);

inline Booleans operator==(double a, const RTensor &b) { return b == a; }
inline Booleans operator<(double a, const RTensor &b) { return b >= a; }
inline Booleans operator>(double a, const RTensor &b) { return b <= a; }
inline Booleans operator<=(double a, const RTensor &b) { return b > a; }
inline Booleans operator>=(double a, const RTensor &b) { return b < a; }
inline Booleans operator!=(double a, const RTensor &b) { return b != a; }

extern template RTensor operator+(const RTensor &a, const RTensor &b);
extern template RTensor operator-(const RTensor &a, const RTensor &b);
extern template RTensor operator-(const RTensor &a);
extern template RTensor operator*(const RTensor &a, const RTensor &b);
extern template RTensor operator/(const RTensor &a, const RTensor &b);

extern template RTensor operator+(const RTensor &a, double b);
extern template RTensor operator-(const RTensor &a, double b);
extern template RTensor operator*(const RTensor &a, double b);
extern template RTensor operator/(const RTensor &a, double b);

extern template RTensor operator+(double a, const RTensor &b);
extern template RTensor operator-(double a, const RTensor &b);
extern template RTensor operator*(double a, const RTensor &b);
extern template RTensor operator/(double a, const RTensor &b);

extern template RTensor &operator+=(RTensor &a, const RTensor &b);
extern template RTensor &operator-=(RTensor &a, const RTensor &b);
extern template RTensor &operator*=(RTensor &a, const RTensor &b);

extern template RTensor &operator+=(RTensor &a, double b);
extern template RTensor &operator-=(RTensor &a, double b);
extern template RTensor &operator*=(RTensor &a, double b);

RTensor kron(const RTensor &a, const RTensor &b);
RTensor kron2(const RTensor &a, const RTensor &b);
RTensor kron2_sum(const RTensor &a, const RTensor &b);

/** Convert a vector of indices to a 1D tensor of real numbers.*/
RTensor index_to_tensor(const Indices &i);

}  // namespace tensor

/* @} */

#endif  // TENSOR_TENSOR_RTENSOR_H