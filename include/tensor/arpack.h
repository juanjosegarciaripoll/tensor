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

#pragma once
#ifndef TENSOR_ARPACK_H
#define TENSOR_ARPACK_H

#include <tensor/arpack_d.h>
#include <tensor/arpack_z.h>

namespace linalg {

template <class Tensor>
class Arpack;
template <>
class Arpack<RTensor> : public RArpack {
 public:
  Arpack(size_t n, enum EigType t, size_t neig) : RArpack(n, t, neig) {}
};

template <>
class Arpack<CTensor> : public CArpack {
 public:
  Arpack(size_t n, enum EigType t, size_t neig) : CArpack(n, t, neig) {}
};

}  // namespace linalg

#endif  // TENSOR_ARPACK_H
