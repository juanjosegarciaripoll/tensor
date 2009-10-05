// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <iostream>
#include <tensor/indices.h>

namespace tensor {

  index
  Range::size(index dim) const
  {
    if (dim == 0)
      return 0;
    switch (type) {
    case Full: {
      return dim;
    }
    case Stepped: {
      index k0 = i0 < 0? dim + i0 : i0;
      index k1 = i1 < 0? dim + i1 : i1;
      if (k1 >= dim || k0 >= dim) {
        std::cerr << "In Range(): The interval [" << k0 << "," << k1
                  << "] exceeds the limits of the\nselected index, which go from 0 to " << (dim-1);
        abort();
      }
      return ((k1 - k0)/di + 1);
    }
    case List: {
      return v.size();
    }
    default:
      std::cerr << "Not a valid Range type " << type << '\n';
      abort();
      return 0;
    }
  }

  void
  Range::to_offsets(Indices &offsets, index increment, index dimension) const
  {
    index k0, k1, dk;
    index l = size(dimension);
    offsets.resize(l);
    switch (type) {
    case Full:
      for (k0 = 0, k1 = 0; k0 < dimension; k0++) {
        offsets.at(k0) = k1;
        k1 += increment;
      }
      break;
    case Stepped:
      k0 = i0 < 0? dimension + i0 : i0;
      k1 = i1 < 0? dimension + i1 : i1;
      if (k1 >= dimension || k0 >= dimension) {
        std::cerr << "In Range(): The interval [" << k0 << "," << k1
                  << "] exceeds the limits of the\nselected index, which go from 0 to "
                  << (dimension-1) << std::endl;
        abort();
      }
      dk = di;
      for (index j = 0; k0 <= k1; k0+=dk, j++) {
        offsets.at(j) = k0 * increment;
      }
      break;
    case List:
      for (index j = 0; j < l; j++) {
        index ndx = v[j];
        if (ndx >= dimension) {
          std::cerr << "In Range(V), some of the elements of the index vector V exceed the\n"
                    << "limits of the selected index, which go from 0 to " << dimension
                    << std::endl;
          abort();
        }
        offsets.at(j) = ndx * increment;
      }
      break;
    default:
      std::cerr << "Not a valid Range type " << type << '\n';
      abort();
    }
  }

} // namespace tensor
