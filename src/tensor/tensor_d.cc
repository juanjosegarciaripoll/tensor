// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#define LOAD_TENSOR_IMPL
#include <tensor/tensor.h>

namespace tensor {

//
// Explicitely instantiate an specialization of Tensor. This generates
// all required code.
//
template class Tensor<double>;

} // namespace
