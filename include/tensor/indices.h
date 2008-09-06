// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_INDICES_H
#define TENSOR_INDICES_H

#include <list>
#include <tensor/vector.h>

namespace tensor {

class ListGenerator {};

extern ListGenerator gen;

template<typename elt_t>
std::list<elt_t> &operator<<(std::list<elt_t> &l, const elt_t &x) {
  l.push_back(x);
  return l;
}

template<typename elt_t>
std::list<elt_t> operator<<(const ListGenerator &g, const elt_t &x) {
  std::list<elt_t> output;
  output.push_back(x);
  return output;
}

typedef long index;

typedef Vector<index> Indices;

extern template class Vector<index>;

}; // namespace

#endif // !TENSOR_H
