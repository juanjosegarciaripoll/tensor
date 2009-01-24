// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_LINALG_H
#define TENSOR_LINALG_H

#include <tensor/tensor.h>

namespace linalg {

  using tensor::RTensor;
  using tensor::CTensor;

  extern bool accurate_svd;

  RTensor svd(RTensor A, RTensor *pU = 0, RTensor *pVT = 0, bool economic = 0);
  RTensor svd(CTensor A, CTensor *pU = 0, CTensor *pVT = 0, bool economic = 0);

} // namespace linalg

#endif // !TENSOR_LINALG_H
