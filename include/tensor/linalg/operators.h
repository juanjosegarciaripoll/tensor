#pragma once
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
#ifndef TENSOR_LINALG_OPERATORS_H
#define TENSOR_LINALG_OPERATORS_H

#include <functional>
#include <tensor/tensor.h>
#include <tensor/sparse.h>

namespace linalg {

using tensor::CSparse;
using tensor::CTensor;
using tensor::RSparse;
using tensor::RTensor;


/** Template for functions that transform tensors into tensors linearly. */
template <typename Tensor>
using LinearMap = std::function<Tensor(const Tensor &)>;

/** Templates for functions that transform tensors into tensors, modifying the input tensor.*/
template <typename Tensor>
using InPlaceLinearMap =
    std::function<void(const Tensor &input, Tensor &output)>;

}

#endif // TENSOR_LINALG_OPERATORS_H
