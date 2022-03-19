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

#include <tensor/rand.h>

namespace tensor {

default_rng_t &default_rng() {
  static default_rng_t default_rng_{};
  return default_rng_;
}

void set_seed(unsigned long seed) {
  default_rng().seed(static_cast<default_rng_t::result_type>(seed));
}

void rand_reseed() {
  std::random_device rd;
  default_rng().seed(rd());
}

}  // namespace tensor
