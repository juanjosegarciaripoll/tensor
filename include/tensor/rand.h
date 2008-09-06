// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_RAND_H
#define TENSOR_RAND_H

#include <tensor/numbers.h>

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

template<class number> inline number rand(number upper_limit) {
  return upper_limit * rand<number>();
}

template<class number> inline number rand(number upper_limit,
                                          number lower_limit) {
  return rand(upper_limit - lower_limit) + lower_limit;
}

#endif // !TENSOR_RAND_H
