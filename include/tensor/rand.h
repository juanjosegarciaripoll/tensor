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

#ifndef TENSOR_RAND_H
#define TENSOR_RAND_H

#include <limits>
#include <type_traits>
#include <random>
#include <tensor/numbers.h>

namespace tensor {

/** Reset the random number generator. If the environment variable
    RANDSEED is defined, that value is used. Otherwise we try to
    gather enough random numbers, either from /dev/urandom, from
    the clock or from other sources. */
void rand_reseed();

/** Reseeds the random number generator with the given value. */
void set_seed(unsigned long seed);

#ifdef TENSOR_64BITS
using default_rng_t = std::mt19937;
#else
using default_rng_t = std::mt19937_64;
#endif

extern default_rng_t &default_rng();

namespace detail {

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
constexpr T rand_inner(T min, T max) {
  if (max > min + 1) {
    std::uniform_int_distribution<T> dist(min, max - 1);
    return dist(default_rng());
  } else {
    return min;
  }
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
constexpr T rand_inner(T min, T max) {
  std::uniform_real_distribution<T> dist(min, max);
  return dist(default_rng());
}

constexpr cdouble rand_inner(cdouble min, cdouble max) {
  return cdouble(rand_inner(min.real(), max.real()),
                 rand_inner(min.imag(), max.imag()));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value, bool> = true>
constexpr T rand_upper_limit() {
  return T(1);
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
constexpr T rand_upper_limit() {
  return T(std::numeric_limits<T>::max());
}

template <>
constexpr cdouble rand_upper_limit<cdouble>() {
  return cdouble(1.0, 1.0);
}

template <typename T>
constexpr T rand_lower_limit() {
  return T(0);
}

}  // namespace detail

/** Returns a random number of the given T. If T is an integer type,
	the value lays in the range [0, max), excluding `max`. If T is a
	complex type, the real and imaginary parts are random numbers
	created with the real and imaginary parts of `max`.
*/
template <typename T>
T rand(T max = detail::rand_upper_limit<T>()) {
  return detail::rand_inner(detail::rand_lower_limit<T>(), max);
}

/** Returns a random number of the given T. If T is an integer or
	floating point type, the value lays in the range [min, max),
	excluding `max`. If T is a complex type, the real and imaginary
	parts are random numbers created with the real and imaginary parts
	of `max`.
*/
template <typename T>
T rand(T lower_bound, T upper_bound) {
  return detail::rand_inner(lower_bound, upper_bound);
}

}  // namespace tensor

#endif  // !TENSOR_RAND_H
