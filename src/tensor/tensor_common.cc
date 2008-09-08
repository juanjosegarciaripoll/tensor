// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#define TENSOR_LOAD_IMPL
#include <numeric>
#include <functional>
#include <iostream>
#include <tensor/tensor.h>

namespace tensor {

bool verify_tensor_dims(const Indices &d, index total_size) {
  for (Indices::const_iterator it = d.begin(); it != d.end(); ++it) {
    if (*it < 0) {
      std::cerr << "Negative dimension in tensor's dimension #"
		<< (it - d.begin()) << std::endl;
      abort();
    }
    total_size /= *it;
    if (total_size <= 0) {
      std::cerr << "Product of tensor dimensions exceeds index range"
		<< std::endl;
      abort();
    }
  }
  return true;
}

index multiply_dimensions(const Indices &d) {
  return std::accumulate(d.begin_const(), d.end_const(),
			 static_cast<index>(0), std::multiplies<index>());
}

} // namespace
