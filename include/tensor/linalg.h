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
#ifndef TENSOR_LINALG_H
#define TENSOR_LINALG_H

#include <functional>
#include <tensor/tensor.h>
#include <tensor/sparse.h>

/*!\addtogroup Linalg*/
/** Namespace for Linear Algebra functions based on BLAS, LAPACK, Lanczos and related algorithms. */
namespace linalg {

using tensor::CSparse;
using tensor::CTensor;
using tensor::RSparse;
using tensor::RTensor;

/** Template for functions that transform tensors into tensors linearly. */
template <typename Tensor>
using LinearMap = std::function<Tensor(const Tensor &)>;

RTensor solve(const RTensor &A, const RTensor &B);
CTensor solve(const CTensor &A, const CTensor &B);

RTensor solve_with_svd(const RTensor &A, const RTensor &B, double tol = 0.0);
CTensor solve_with_svd(const CTensor &A, const CTensor &B, double tol = 0.0);

/**Solve a real linear system of equations by the conjugate gradient method. 'f'
 * is a linear map between tensors, acting similar to multiplication by a matrix.*/
RTensor cgs(const LinearMap<RTensor> &f, const RTensor &b,
            const RTensor *x_start = 0, int maxiter = 0, double tol = 0);

/**Solve a real linear system of equations by the conjugate gradient method.  'f'
 * is a linear map between tensors, acting similar to multiplication by a matrix.*/
CTensor cgs(const LinearMap<CTensor> &f, const CTensor &b,
            const CTensor *x_start = 0, int maxiter = 0, double tol = 0);

/**Solve a real linear system of equations by the conjugate gradient method.*/
RTensor cgs(const RTensor &A, const RTensor &b, const RTensor *x_start = 0,
            int maxiter = 0, double tol = 0);
/**Solve a real linear system of equations by the conjugate gradient method.*/
CTensor cgs(const CTensor &A, const CTensor &b, const CTensor *x_start = 0,
            int maxiter = 0, double tol = 0);

/**Solve a real linear system of equations by the conjugate gradient method.*/
RTensor cgs(const RSparse &A, const RTensor &b, const RTensor *x_start = 0,
            int maxiter = 0, double tol = 0);
/**Solve a real linear system of equations by the conjugate gradient method.*/
CTensor cgs(const RSparse &A, const CTensor &b, const CTensor *x_start = 0,
            int maxiter = 0, double tol = 0);

extern bool accurate_svd;

#define SVD_ECONOMIC true
RTensor svd(RTensor A, RTensor *pU = 0, RTensor *pVT = 0, bool economic = 0);
RTensor svd(CTensor A, CTensor *pU = 0, CTensor *pVT = 0, bool economic = 0);

RTensor block_svd(RTensor A, RTensor *pU = 0, RTensor *pVT = 0,
                  bool economic = 0);
RTensor block_svd(CTensor A, CTensor *pU = 0, CTensor *pVT = 0,
                  bool economic = 0);

/**Eigenvalue decomposition of a real matrix.*/
CTensor eig(const RTensor &A, CTensor *R = 0, CTensor *L = 0);

/**Eigenvalue decomposition of a complex matrix.*/
CTensor eig(const CTensor &A, CTensor *R = 0, CTensor *L = 0);

/**Compute the right eigenvector with the largest absolute eigenvalue using the
     power method.*/
double eig_power_right(const RTensor &A, RTensor *vector, size_t iter = 0,
                       double tol = 1e-11);
/**Compute the left eigenvector with the largest absolute eigenvalue using the
     power method.*/
double eig_power_left(const RTensor &A, RTensor *vector, size_t iter = 0,
                      double tol = 1e-11);
/**Compute the right eigenvector with the largest absolute eigenvalue using the
     power method.*/
double eig_power_right(const RSparse &A, RTensor *vector, size_t iter = 0,
                       double tol = 1e-11);
/**Compute the left eigenvector with the largest absolute eigenvalue using the
     power method.*/
double eig_power_left(const RSparse &A, RTensor *vector, size_t iter = 0,
                      double tol = 1e-11);
/**Compute the right eigenvector with the largest absolute eigenvalue using the
     power method.*/
tensor::cdouble eig_power_right(const CTensor &A, CTensor *vector,
                                size_t iter = 0, double tol = 1e-11);
/**Compute the left eigenvector with the largest absolute eigenvalue using the
     power method.*/
tensor::cdouble eig_power_left(const CTensor &A, CTensor *vector,
                               size_t iter = 0, double tol = 1e-11);
/**Compute the right eigenvector with the largest absolute eigenvalue using the
     power method.*/
tensor::cdouble eig_power_right(const CSparse &A, CTensor *vector,
                                size_t iter = 0, double tol = 1e-11);
/**Compute the left eigenvector with the largest absolute eigenvalue using the
     power method.*/
tensor::cdouble eig_power_left(const CSparse &A, CTensor *vector,
                               size_t iter = 0, double tol = 1e-11);

/**Compute the right eigenvector with the largest absolute eigenvalue using the
     power method. 'f' is a function that takes in a Tensor and returns also a
     Tensor of the same class and dimension. */
double eig_power(const LinearMap<RTensor> &f, size_t dim, RTensor *vector,
                 size_t iter = 0, double tol = 1e-11);

/**Compute the right eigenvector with the largest absolute eigenvalue using the
     power method. 'f' is a function that takes in a Tensor and returns also a
     Tensor of the same class and dimension. */
tensor::cdouble eig_power(const LinearMap<CTensor> &f, size_t dim,
                          CTensor *vector, size_t iter = 0, double tol = 1e-11);

RTensor eig_sym(const RTensor &A, RTensor *pR = 0);
RTensor eig_sym(const CTensor &A, CTensor *pR = 0);

RTensor expm(const RTensor &A, unsigned int order = 7);
CTensor expm(const CTensor &A, unsigned int order = 7);

/**Type of eigenvalues that eigs and Arpack compute.*/
enum EigType {
  LargestMagnitude = 0,  /*!<Eigenvalues with largest modulus.*/
  SmallestMagnitude = 1, /*!<Eigenvalues with smallest modulus.*/
  LargestReal = 2,       /*!<Eigenvalues with largest real part.*/
  LargestAlgebraic = 2,  /*!<Eigenvalues with largest real part.*/
  SmallestReal = 3,      /*!<Eigenvalues with smallest real part.*/
  SmallestAlgebraic = 3, /*!<Eigenvalues with smallest real part.*/
  LargestImaginary = 4,  /*!<Eigenvalues with the largest imaginary part.*/
  SmallestImaginary = 5  /*!<Eigenvalues with the smallest imaginary part.*/
};

/**Find out a few eigenvalues and eigenvectors of a complex nonsymmetric
     matrix. 'vectors' is used to output the eigenvectors, but it can also
     contain a good estimate of them. 'converged' is true when the algorithm
     finished properly. */
CTensor eigs(const CTensor &A, int eig_type, size_t neig,
             CTensor *vectors = NULL, bool *converged = NULL);

/**Find out a few eigenvalues and eigenvectors of a nonsymmetric complex
     sparse matrix. 'vectors' is used to output the eigenvectors, but it can
     also contain a good estimate of them. 'converged' is true when the
     algorithm finished properly. */
CTensor eigs(const CSparse &A, int eig_type, size_t neig,
             CTensor *vectors = NULL, bool *converged = NULL);

/**Find out a few eigenvalues and eigenvectors of a real nonsymmetric
     matrix. 'vectors' is used to output the eigenvectors, but it can also
     contain a good estimate of them. 'converged' is true when the algorithm
     finished properly. */
RTensor eigs(const RTensor &A, int eig_type, size_t neig,
             RTensor *vectors = NULL, bool *converged = NULL);

/**Find out a few eigenvalues and eigenvectors of a nonsymmetric real sparse
     matrix. 'vectors' is used to output the eigenvectors, but it can also
     contain a good estimate of them. 'converged' is true when the algorithm
     finished properly. */
RTensor eigs(const RSparse &A, int eig_type, size_t neig,
             RTensor *vectors = NULL, bool *converged = NULL);

/**Find out a few eigenvalues and eigenvectors of a nonsymmetric real sparse
     matrix. 'f' is a function that takes in a Tensor and returns also a Tensor
     of the same class and dimension. Because we do not know the dimensions of
     'f', this has to be provided in 'dim'. 'vectors' is used to output the
     eigenvectors, but it can also contain a good estimate of them. 'converged'
     is true when the algorithm finished properly. */
RTensor eigs(const LinearMap<RTensor> &f, size_t dim, int eig_type, size_t neig,
             RTensor *vectors = NULL, bool *converged = NULL);

/**Find out a few eigenvalues and eigenvectors of a nonsymmetric real sparse
     matrix. 'f' is a function that takes in a Tensor and returns also a Tensor
     of the same class and dimension. Because we do not know the dimensions of
     'f', this has to be provided in 'dim'. 'vectors' is used to output the
     eigenvectors, but it can also contain a good estimate of them. 'converged'
     is true when the algorithm finished properly. */
CTensor eigs(const LinearMap<CTensor> &f, size_t dim, int eig_type, size_t neig,
             CTensor *vectors = NULL, bool *converged = NULL);

/**Find out a few eigenvalues and eigenvectors of a symmetric real matrix.*/
RTensor eigs_sym(const RTensor &A, int eig_type, size_t neig,
                 RTensor *vectors = NULL, bool *converged = NULL);

/**Find out a few eigenvalues and eigenvectors of a symmetric real matrix.*/
RTensor eigs_sym(const CTensor &A, int eig_type, size_t neig,
                 CTensor *vectors = NULL, bool *converged = NULL);

/**Find out a few eigenvalues and eigenvectors of a symmetric real sparse matrix.*/
RTensor eigs_sym(const RSparse &A, int eig_type, size_t neig,
                 RTensor *vectors = NULL, bool *converged = NULL);

/**Find out a few eigenvalues and eigenvectors of a hermitian complex sparse matrix.*/
RTensor eigs_sym(const CSparse &A, int eig_type, size_t neig,
                 CTensor *vectors = NULL, bool *converged = NULL);

}  // namespace linalg

#endif  // !TENSOR_LINALG_H
