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

#pragma once
#ifndef TENSOR_NUMBERS_H
#define TENSOR_NUMBERS_H

#include <cmath>
#include <complex>

namespace tensor {

using std::abs;

typedef std::ptrdiff_t index;

//
// REAL NUMBERS
//

template <typename number>
inline number number_zero() {
  return static_cast<number>(0);
}

template <typename number>
inline number number_one() {
  return static_cast<number>(1);
}

/* Already in C++11 but they return complex */
inline double real(double r) { return r; }
inline double imag(double) { return 0.0; }
inline double conj(double r) { return r; }
inline double abs2(double r) { return r * r; }

template <class number>
inline number square(number r) {
  return r * r;
}

//
// COMPLEX NUMBERS
//

typedef std::complex<double> cdouble;

inline cdouble to_complex(const double &r) { return cdouble(r, 0); }

inline cdouble to_complex(const double &r, const double &i) {
  return cdouble(r, i);
}

inline cdouble to_complex(const cdouble &z) { return z; }

template <>
inline cdouble number_zero<cdouble>() {
  return to_complex(0.0);
}

template <>
inline cdouble number_one<cdouble>() {
  return to_complex(1.0);
}

inline double real(cdouble z) { return std::real(z); }
inline double imag(cdouble z) { return std::imag(z); }
inline cdouble conj(cdouble z) { return std::conj(z); }
inline double abs2(cdouble z) { return abs2(real(z)) + abs2(imag(z)); }

#if defined(_MSC_VER) && (_MSC_VER < 1800)
double round(double r);
inline cdouble round(cdouble r) {
  return to_complex(round(real(r)), round(imag(r)));
};
#else
inline cdouble round(cdouble r) {
  return to_complex(::round(real(r)), ::round(imag(r)));
}
#endif

}  // namespace tensor

#endif  // !TENSOR_NUMBERS_H
