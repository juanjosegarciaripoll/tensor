// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_CLAPACK_H
#define TENSOR_CLAPACK_H

#include <tensor/cblas.h>

namespace lapack {

  using namespace blas;

#ifdef __APPLE__
#include <vecLib/clapack.h>
#endif

}

#endif // TENSOR_CLAPACK_H
