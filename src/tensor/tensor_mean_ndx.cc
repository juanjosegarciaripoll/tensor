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

#include <tensor/tensor.h>

namespace {

using namespace tensor;

template <typename Tensor>
Tensor do_mean(const Tensor &t, int ndx) {
  typedef typename Tensor::elt_t elt_t;
  int rank = t.rank();
  if (rank == 1) {
    return Tensor(Dimensions{1}, Vector<elt_t>({mean(t)}));
  } else {
    ndx = (int)Dimensions::normalize_index(ndx, rank);
    Indices dimensions(rank - 1);
    tensor::index left = 1, middle = 1, right = 1;
    for (int i = 0, j = 0; i < rank; i++) {
      tensor::index d = t.dimension(i);
      dimensions.at(j++) = d;
      if (i < ndx)
        left *= d;
      else if (i > ndx)
        right *= d;
      else {
        middle = d;
        j--;
      }
    }
    Tensor aux = reshape(t, left, middle, right);
    Tensor output = Tensor::zeros(left, right);
    for (tensor::index r = 0; r < right; r++)
      for (tensor::index m = 0; m < middle; m++)
        for (tensor::index l = 0; l < left; l++)
          output.at(l, r) += aux(l, m, r);
    return reshape(output, dimensions) / static_cast<double>(middle);
  }
}

}  // namespace
