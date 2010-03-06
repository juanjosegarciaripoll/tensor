// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <numeric>
#include <functional>
#include <tensor/indices.h>

namespace tensor {

  const ListGenerator<index> igen = {};
  const ListGenerator<double> rgen = {};
  const ListGenerator<cdouble> cgen = {};

  template class Vector<index>;

  bool Indices::operator==(const Indices &other) const {
    return std::equal(other.begin_const(), other.end_const(), this->begin_const());
  }

  index Indices::total_size() const {
    if (size()) {
      return std::accumulate(begin_const(), end_const(),
                             static_cast<index>(1), std::multiplies<index>());
    } else {
      return 0;
    }
  }


} // namespace
