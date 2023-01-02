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

#define TENSOR_LOAD_IMPL
#include <tensor/tensor.h>
#include <tensor/tensor/implementation.h>

//
// Explicitely instantiate an specialization of Tensor. This generates
// all required code.
//
template class tensor::Tensor<tensor::cdouble>;
template class tensor::TensorView<tensor::cdouble>;
template class tensor::MutableTensorView<tensor::cdouble>;
