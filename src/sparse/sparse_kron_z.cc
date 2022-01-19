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

#include <tensor/sparse.h>
#include "sparse_kron.hpp"

namespace tensor {

// Explicit instantiation
CSparse kron(const CSparse &s1, const CSparse &s2) { return do_kron(s1, s2); }

CSparse kron2(const CSparse &s1, const CSparse &s2) { return kron(s2, s1); }

CSparse kron2_sum(const CSparse &s2, const CSparse &s1) {
  return kron(s1, CSparse::eye(s2.length())) +
         kron(CSparse::eye(s1.length()), s2);
}

}  // namespace tensor
