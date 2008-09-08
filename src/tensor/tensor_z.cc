// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#define TENSOR_LOAD_IMPL
#include <tensor/tensor.h>

//
// Explicitely instantiate an specialization of Tensor. This generates
// all required code.
//
template class tensor::Tensor<tensor::cdouble>;
