// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_RAND_H
#define TENSOR_RAND_H

#include <tensor/numbers.h>

void rand_reseed();

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
