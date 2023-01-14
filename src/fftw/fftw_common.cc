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

#include <vector>
#include <limits>
#include "fftw_common.hpp"

namespace tensor {

using std::numeric_limits;
using std::vector;

int inline index_to_fftw_dimension(const Indices& dims, int which) {
  auto d = dims[which];
  if (d > numeric_limits<int>::max()) {
    std::cerr << "Dimension " << which << " of array exceed maximum FFTW size "
              << numeric_limits<int>::max() << std::endl;
    abort();
  }
  return static_cast<int>(d);
}

// basic FFTW along all degrees of freedom
void do_fftw(fftw_complex* pin, fftw_complex* pout, const Indices& dims,
             int direction) {
  fftw_plan plan{};
  int d = static_cast<int>(dims.ssize());

  /** \todo Rewrite this in separate functions */
  if (d == 1) {
    plan = fftw_plan_dft_1d(index_to_fftw_dimension(dims, 0), pin, pout,
                            direction, FFTW_ESTIMATE);
  } else {
    auto dimensions = SimpleVector<int>::empty(static_cast<size_t>(d));
    for (int i = 0; i < d; i++) {
      dimensions.at(d - i - 1) = index_to_fftw_dimension(dims, i);
    }
    plan = fftw_plan_dft(d, dimensions.begin(), pin, pout, direction,
                         FFTW_ESTIMATE);
  }

  fftw_execute(plan);
  fftw_destroy_plan(plan);
}

void do_fftw(fftw_complex* pin, fftw_complex* pout, int dim,
             const Indices& dims, int direction) {
  // Collect the dimensions for the transform
  fftw_iodim fft_dim;
  auto loop_dims = SimpleVector<fftw_iodim>::empty(dims.size() - 1);
  int stride = 1;

  for (int i = 0; i < dim; i++) {
    loop_dims.at(i).n = index_to_fftw_dimension(dims, i);
    loop_dims.at(i).is = stride;
    loop_dims.at(i).os = stride;
    stride *= loop_dims.at(i).n;
  }

  fft_dim.n = index_to_fftw_dimension(dims, dim);
  fft_dim.is = stride;
  fft_dim.os = stride;
  stride *= fft_dim.n;

  for (int i = dim + 1; i < dims.ssize(); i++) {
    loop_dims.at(i - 1).n = index_to_fftw_dimension(dims, i);
    loop_dims.at(i - 1).is = stride;
    loop_dims.at(i - 1).os = stride;
    stride *= loop_dims.at(i - 1).n;
  }

  // Now make a Guru plan and execute it.
  fftw_plan plan = fftw_plan_guru_dft(
      1, &fft_dim, static_cast<int>(dims.size()) - 1, loop_dims.begin(), pin,
      pout, direction, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);
}

void do_fftw(fftw_complex* pin, fftw_complex* pout, const Booleans& convert,
             const Indices& dims, int direction) {
  // Collect the dimensions for the transform
  auto fft_dims = SimpleVector<fftw_iodim>::empty(dims.size());
  auto loop_dims = SimpleVector<fftw_iodim>::empty(dims.size());
  int fft_size = 0;
  int loop_size = 0;
  int stride = 1;

  for (int i = 0; i < dims.ssize(); i++) {
    int d = index_to_fftw_dimension(dims, i);

    if (convert[i] == false) {
      loop_dims.at(loop_size).n = d;
      loop_dims.at(loop_size).is = stride;
      loop_dims.at(loop_size).os = stride;
      loop_size++;
    } else {
      fft_dims.at(fft_size).n = d;
      fft_dims.at(fft_size).is = stride;
      fft_dims.at(fft_size).os = stride;
      fft_size++;
    }

    stride *= d;
  }

  // Now make a Guru plan and execute it.
  fftw_plan plan = fftw_plan_guru_dft(fft_size, fft_dims.begin(), loop_size,
                                      loop_dims.begin(), pin, pout, direction,
                                      FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);
}

}  // namespace tensor
