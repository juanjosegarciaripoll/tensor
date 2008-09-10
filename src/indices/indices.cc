// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <tensor/indices.h>

namespace tensor {

template class Vector<index>;

bool Indices::operator==(const Indices &other) const {
  return std::equal(other.begin_const(), other.end_const(), this->begin_const());
}

} // namespace
