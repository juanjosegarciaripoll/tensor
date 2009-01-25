// Copyright 2008, Juan Jose Garcia-Ripoll
//

#define TENSOR_LOAD_IMPL
#include <tensor/tensor.h>

namespace tensor {

  double norm0(const CTensor &r)
  {
    double output = 0;
    for (CTensor::const_iterator it = r.begin(); it != r.end(); ++it) {
      output = std::max(output, abs(*it));
    }
    return output;
  }

} // namespace tensor
