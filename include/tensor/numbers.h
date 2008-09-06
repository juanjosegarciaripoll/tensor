// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_NUMBERS_H
#define TENSOR_NUMBERS_H

#include <cmath>
#include <complex>
#include <iostream>

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

#endif // !TENSOR_NUMBERS_H
