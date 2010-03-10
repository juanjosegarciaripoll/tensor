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

#ifndef TENSOR_NUMBERS_H
#define TENSOR_NUMBERS_H

#include <cmath>
#include <complex>
#include <iostream>

namespace tensor {

//
// REAL NUMBERS
//

template<typename number>
inline number number_zero() { return static_cast<number>(0); }

template<typename number>
inline number number_one() { return static_cast<number>(1); }

inline double real(double r) { return r; }

inline double imag(double r) { return 0.0; }

inline double conj(double r) { return r; }

inline double abs2(double r) { return r*r; }

template<class number> number square(number r) { return r*r; }

//
// COMPLEX NUMBERS
//

typedef std::complex<double> cdouble;

inline cdouble to_complex(const double &r, const double &i = 0.0) {
  return cdouble(r, i);
}

inline cdouble to_complex(const cdouble &z) {
  return z;
}

template<>
inline cdouble number_zero<cdouble>() { return to_complex(0.0); }

template<>
inline cdouble number_one<cdouble>() { return to_complex(1.0); }

using std::real;
using std::imag;
using std::conj;
using std::abs;

inline double abs2(cdouble z) { return abs2(real(z)) + abs2(imag(z)); }

inline cdouble round(cdouble r) {
  return to_complex(::round(real(r)),::round(imag(r)));
}

inline std::istream &operator>>(std::istream &s, cdouble &z) {
  double r, i;
  s >> r >> i;
  z = to_complex(r, i);
  return s;
}

inline std::ostream &operator<<(std::ostream &s, const cdouble &d) {
  return s << real(d) << ' ' << imag(d);
}

} // namespace tensor

#endif // !TENSOR_NUMBERS_H
