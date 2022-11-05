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
#ifndef TENSOR_TENSOR_CTENSOR_H
#define TENSOR_TENSOR_CTENSOR_H

#include <tensor/tensor/types.h>
#include <tensor/tensor/operators.h>

/*!\addtogroup Tensors*/
/* @{ */

namespace tensor {

CTensor change_dimension(const CTensor &a, int dimension, index new_size);

/**Return the sum of the elements in the tensor.*/
cdouble sum(const CTensor &r);
/**Return the mean of the elements in the tensor.*/
cdouble mean(const CTensor &r);
/**Return the mean of the elements in the along the given dimension.*/
extern template CTensor mean(const CTensor &t, index ndx);

double norm0(const CTensor &r);
cdouble scprod(const CTensor &a, const CTensor &b);
double norm2(const CTensor &r);
double matrix_norminf(const CTensor &m);

inline RTensor real(const RTensor &r) { return r; }
RTensor imag(const RTensor &t);
RTensor real(const CTensor &t);
RTensor imag(const CTensor &t);

CTensor to_complex(const RTensor &r);
inline const CTensor &to_complex(const CTensor &r) { return r; }
CTensor to_complex(const RTensor &r, const RTensor &i);

/**Complex conjugate of a real tensor. Returns the same tensor.*/
inline const RTensor &conj(const RTensor &r) { return r; }
const CTensor conj(const CTensor &t);

RTensor abs(const CTensor &t);
CTensor cos(const CTensor &t);
CTensor sin(const CTensor &t);
CTensor tan(const CTensor &t);
CTensor cosh(const CTensor &t);
CTensor sinh(const CTensor &t);
CTensor tanh(const CTensor &t);
CTensor exp(const CTensor &t);
CTensor sqrt(const CTensor &t);
CTensor log(const CTensor &t);
CTensor pow(const CTensor &base, double expt);
CTensor pow(const CTensor &base, cdouble expt);

CTensor diag(const CTensor &a, int which, index rows, index cols);
CTensor diag(const CTensor &a, int which = 0);
CTensor take_diag(const CTensor &a, int which = 0, int ndx1 = 0, int ndx2 = -1);
cdouble trace(const CTensor &a);
CTensor trace(const CTensor &a, int ndx1, int ndx2);

CTensor squeeze(const CTensor &t);
CTensor permute(const CTensor &a, index ndx1 = 0, index ndx2 = -1);
CTensor transpose(const CTensor &a);
CTensor adjoint(const CTensor &a);

CTensor fold(const CTensor &a, int ndx1, const CTensor &b, int ndx2);
CTensor fold(const RTensor &a, int ndx1, const CTensor &b, int ndx2);
CTensor fold(const CTensor &a, int ndx1, const RTensor &b, int ndx2);

CTensor foldc(const CTensor &a, int ndx1, const CTensor &b, int ndx2);
CTensor foldc(const RTensor &a, int ndx1, const CTensor &b, int ndx2);
CTensor foldc(const CTensor &a, int ndx1, const RTensor &b, int ndx2);

CTensor mmult(const CTensor &a, const CTensor &b);
CTensor mmult(const RTensor &a, const CTensor &b);
CTensor mmult(const CTensor &a, const RTensor &b);

CTensor scale(const CTensor &t, int ndx, const CTensor &v);
CTensor scale(const CTensor &t, int ndx, const RTensor &v);
void scale_inplace(CTensor &t, int ndx, const CTensor &v);
void scale_inplace(CTensor &t, int ndx, const RTensor &v);

CTensor foldin(const CTensor &a, int ndx1, const CTensor &b, int ndx2);

CTensor linspace(cdouble min, cdouble max, index n = 100);
CTensor linspace(const CTensor &min, const CTensor &max, index n = 100);

CTensor sort(const CTensor &v, bool reverse = false);
Indices sort_indices(const CTensor &v, bool reverse = false);

extern template bool all_equal(const CTensor &a, const CTensor &b);
extern template bool all_equal(const CTensor &a, cdouble b);

extern template Booleans operator==(const CTensor &a, const CTensor &b);
extern template Booleans operator!=(const CTensor &a, const CTensor &b);
extern template Booleans operator==(const CTensor &a, cdouble b);
extern template Booleans operator!=(const CTensor &a, cdouble b);

extern template CTensor operator+(const CTensor &a, const CTensor &b);
extern template CTensor operator-(const CTensor &a, const CTensor &b);
extern template CTensor operator-(const CTensor &a);
extern template CTensor operator*(const CTensor &a, const CTensor &b);
extern template CTensor operator/(const CTensor &a, const CTensor &b);

extern template CTensor operator+(const CTensor &a, cdouble b);
extern template CTensor operator-(const CTensor &a, cdouble b);
extern template CTensor operator*(const CTensor &a, cdouble b);
extern template CTensor operator/(const CTensor &a, cdouble b);

extern template CTensor operator+(cdouble a, const CTensor &b);
extern template CTensor operator-(cdouble a, const CTensor &b);
extern template CTensor operator*(cdouble a, const CTensor &b);
extern template CTensor operator/(cdouble a, const CTensor &b);

extern template CTensor &operator+=(CTensor &a, const CTensor &b);
extern template CTensor &operator-=(CTensor &a, const CTensor &b);
extern template CTensor &operator*=(CTensor &a, const CTensor &b);

extern template CTensor &operator+=(CTensor &a, cdouble b);
extern template CTensor &operator-=(CTensor &a, cdouble b);
extern template CTensor &operator*=(CTensor &a, cdouble b);

CTensor kron(const CTensor &a, const CTensor &b);
CTensor kron2(const CTensor &a, const CTensor &b);
CTensor kron2_sum(const CTensor &a, const CTensor &b);

}  // namespace tensor

/* @} */

#endif  // TENSOR_TENSOR_CTENSOR_H
