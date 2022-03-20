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

#include <tensor/sparse.h>
#include "sparse_kron.hpp"

namespace tensor {

// Explicit instantiation
RSparse kron(const RSparse &a, const RSparse &b) { return do_kron(a, b); }

RSparse kron2(const RSparse &a, const RSparse &b) { return kron(b, a); }

RSparse kron2_sum(const RSparse &a, const RSparse &b) {
  return kron(b, RSparse::eye(a.length())) + kron(RSparse::eye(b.length()), a);
}

}  // namespace tensor
