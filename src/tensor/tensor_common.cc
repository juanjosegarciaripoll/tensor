// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <numeric>
#include <functional>
#include <iostream>
#include <tensor/tensor.h>
#include <tensor/io.h>

namespace tensor {

bool verify_tensor_dimensions(const Indices &d, index total_size) {
  index aux = total_size;
  if (aux == 0) {
    for (Indices::const_iterator it = d.begin_const(); it != d.end_const();
	 ++it) {
      if (*it == 0)
	return true;
    }
    std::cerr << "Product of tensor dimensions exceeds index range."
	      << std::endl
	      << "All dimensions: " << d << std::endl
	      << "Expected size: " << total_size << std::endl;
    return false;
  } else {
    for (Indices::const_iterator it = d.begin_const(); it != d.end_const();
	 ++it) {
      if (*it < 0) {
	std::cerr << "Negative dimension in tensor's dimension #"
		  << (it - d.begin()) << std::endl
		  << "All dimensions:" << std::endl
		  << d << std::endl;
	return false;
      }
      aux /= *it;
      if (aux <= 0) {
	std::cerr << "Product of tensor dimensions exceeds index range."
		  << std::endl
		  << "All dimensions: " << d << std::endl
		  << "Expected size: " << total_size << std::endl;
	return false;
      }
    }
    return true;
  }
}

index multiply_indices(const Indices &d) {
  return std::accumulate(d.begin_const(), d.end_const(),
			 static_cast<index>(1), std::multiplies<index>());
}

} // namespace
