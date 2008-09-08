// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_IO_H
#define TENSOR_IO_H

#include <iostream>

namespace tensor {

/**Simple text representation of vector.*/
template<typename elt_t>
std::ostream &operator<<(std::ostream &s, const Vector<elt_t> &t);

/**Simple text representation of tensor.*/
template<typename elt_t>
std::ostream &operator<<(std::ostream &s, const Tensor<elt_t> &t);

} // namespace tensor

#include <tensor/detail/io.hpp>

#endif // !TENSOR_IO_H
