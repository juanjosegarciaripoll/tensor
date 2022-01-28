// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2013 Juan Jose Garcia Ripoll

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

#include <tensor/fftw.h>

namespace tensor {

CTensor fftshift(const CTensor& input, int direction) {
  // just forward to the simpler function
  CTensor output(input);
  for (index dim : input.dimensions()) {
    output = fftshift(output, dim, direction);
  }

  return output;
}

CTensor fftshift(const CTensor& input, index dim, int direction) {
  assert(dim >= 0 && dim < input.rank());

  const Indices& size = input.dimensions();
  if (size[dim] == 1) {
    // nothing to do
    return input;
  }

  index before = 1;
  for (index i = 0; i < dim; i++) {
    before *= size[i];
  }

  index after = 1;
  for (index i = dim + 1; i < size.size(); i++) {
    after *= size[i];
  }

  // Transform the tensor into a 3-dimensional form, then use slicing.
  // Not terribly efficient, but we can delay extensive memory handling to
  // until someone needs the speed.
  auto output = CTensor::empty(before, size[dim], after);
  const CTensor reshape_input = reshape(input, output.dimensions());

  // even number of grid points as default => direction has no effect.
  index minfreq = size[dim] / 2;
  index Nhalf = size[dim] / 2;
  index end = size[dim] - 1;

  if (size[dim] % 2 == 1) {
    minfreq = (size[dim] + 1) / 2;
    Nhalf = (size[dim] - 1) / 2;
    if (direction == FFTW_BACKWARD) {
      // undo the effects of the forward shift, requires some creativity
      minfreq--;
      Nhalf++;
    }
  }

  output.at(range(), range(0, Nhalf - 1), range()) =
      reshape_input(range(), range(minfreq, end), range());
  output.at(range(), range(Nhalf, end), range()) =
      reshape_input(range(), range(0, minfreq - 1), range());

  return reshape(output, size);
}

CTensor fftshift(const CTensor& input, const Booleans& convert, int direction) {
  assert(input.rank() == convert.size());

  CTensor output(input);
  for (index dim : input.dimensions()) {
    if (convert[dim]) {
      output = fftshift(output, dim, direction);
    }
  }

  return output;
}
}  // namespace tensor