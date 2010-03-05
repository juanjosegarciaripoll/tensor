// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_RAND_H
#define TENSOR_RAND_H

#include <tensor/numbers.h>

namespace tensor {

/** Reset the random number generator. If the environment variable
    RANDSEED is defined, that value is used. Otherwise we try to
    gather enough random numbers, either from /dev/urandom, from
    the clock or from other sources. */
void rand_reseed();

/** Returns a random number. If the type is an integral one, the range
    is the whole integer range; if the type is a complex or floating point
    type, then the range is a n-dimensional cube with a corner on [0,...,0]
    and extending along the positive axis without reaching value of 1 on
    only direction.
*/
template<class number> number rand() {
  return static_cast<number>(rand<double>());
}

template<> int rand<int>();
template<> unsigned int rand<unsigned int>();
template<> long rand<long>();
template<> unsigned long rand<unsigned long>();
template<> float rand<float>();
template<> double rand<double>();
template<> cdouble rand<cdouble>();

template<class real_number> inline real_number rand(real_number upper_limit) {
  return static_cast<real_number>(upper_limit * rand<double>());
}

template<class real_number> inline real_number rand(real_number lower_limit,
                                          real_number upper_limit) {
  return rand<real_number>(upper_limit - lower_limit) + lower_limit;
}

template<> inline int rand<int>(int upper) {
  return rand<int>() % upper;
}

template<> inline int rand<int>(int lower, int upper) {
  return rand<int>(upper - lower) + lower;
}

template<> inline long rand<long>(long upper) {
  return rand<long>() % upper;
}

template<> inline long rand<long>(long lower, long upper) {
  return rand<long>(upper - lower) + lower;
}

template<> inline unsigned long rand<unsigned long>(unsigned long upper) {
  return rand<unsigned long>() % upper;
}

template<> inline unsigned long rand<unsigned long>(unsigned long lower, unsigned long upper) {
  return rand<unsigned long>(upper - lower) + lower;
}
} // tensor

#endif // !TENSOR_RAND_H
