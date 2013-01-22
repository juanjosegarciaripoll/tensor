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

#ifndef TENSOR_ARPACK_H
#define TENSOR_ARPACK_H

#include <tensor/tensor.h>
#include <tensor/sparse.h>
#include <tensor/arpack_d.h>
#include <tensor/arpack_z.h>

namespace linalg {

  /**Find out a few eigenvalues and eigenvectors of a complex nonsymmetric matrix.*/
  tensor::CTensor eigs(const tensor::CTensor &A, int eig_type, size_t neig,
                       tensor::CTensor *vectors = NULL,
                       const tensor::CTensor::elt_t *initial_guess = NULL);

  /**Find out a few eigenvalues and eigenvectors of a nonsymmetric complex sparse matrix.*/
  tensor::CTensor eigs(const tensor::CSparse &A, int eig_type, size_t neig,
                       tensor::CTensor *vectors = NULL,
                       const tensor::CTensor::elt_t *initial_guess = NULL);

  /**Find out a few eigenvalues and eigenvectors of a real nonsymmetric matrix.*/
  tensor::RTensor eigs(const tensor::RTensor &A, int eig_type, size_t neig,
                       tensor::RTensor *vectors = NULL,
                       const tensor::RTensor::elt_t *initial_guess = NULL);

  /**Find out a few eigenvalues and eigenvectors of a nonsymmetric real sparse matrix.*/
  tensor::RTensor eigs(const tensor::RSparse &A, int eig_type, size_t neig,
                       tensor::RTensor *vectors = NULL,
                       const tensor::RTensor::elt_t *initial_guess = NULL);

  /**Find out a few eigenvalues and eigenvectors of a symmetric real matrix.*/
  tensor::RTensor eigs_sym(const tensor::RTensor &A, int eig_type, size_t neig,
                           tensor::RTensor *vectors = NULL,
                           const tensor::RTensor::elt_t *initial_guess = NULL);

  /**Find out a few eigenvalues and eigenvectors of a symmetric real matrix.*/
  tensor::RTensor eigs_sym(const tensor::CTensor &A, int eig_type, size_t neig,
                           tensor::CTensor *vectors = NULL,
                           const tensor::CTensor::elt_t *initial_guess = NULL);

  /**Find out a few eigenvalues and eigenvectors of a symmetric real sparse matrix.*/
  tensor::RTensor eigs_sym(const tensor::RSparse &A, int eig_type, size_t neig,
                           tensor::RTensor *vectors = NULL,
                           const tensor::RTensor::elt_t *initial_guess = NULL);

  /**Find out a few eigenvalues and eigenvectors of a hermitian complex sparse matrix.*/
  tensor::RTensor eigs_sym(const tensor::CSparse &A, int eig_type, size_t neig,
                           tensor::CTensor *vectors = NULL,
                           const tensor::CTensor::elt_t *initial_guess = NULL);

}

#endif /* !TENSOR_ARPACK_H */
