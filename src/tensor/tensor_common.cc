// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2010 Juan Jose Garcia Ripoll

    Tensor is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public License as published
    by the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Library General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include <iostream>
#include <tensor/tensor.h>
#include <tensor/io.h>

namespace tensor {

bool verify_tensor_dimensions(const Indices &d, index total_size) {
  index aux = total_size;
  if (aux == 0) {
    if (d.size() == 0)
      return true;
    for (Indices::const_iterator it = d.begin_const(); it != d.end_const();
	 ++it) {
      if (*it == 0)
	return true;
    }
    std::cerr << "Product of tensor dimensions exceeds data size."
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

bool verify_tensor_dimensions_match(const Indices &d1, const Indices &d2) {
  if ((d1.size() != d2.size()) || !(d1 == d2)) {
    std::cerr << "A binary operation was attempted among two tensors" << std::endl
              << "with different dimensions:" << std::endl
              << "d1 = " << d1 << std::endl
              << "d2 = " << d2 << std::endl;
    return false;
  }
  return true;
}

} // namespace
