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
#ifndef TENSOR_LINALG_EIGS_H
#define TENSOR_LINALG_EIGS_H

#include <tensor/tensor.h>
#include <tensor/linalg/operators.h>

namespace linalg {

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

#ifdef TENSOR_USE_ARPACK
namespace arpack {

static constexpr int min_arpack_size = 10;

RTensor make_matrix(const InPlaceLinearMap<RTensor> &A, size_t n);

CTensor make_matrix(const InPlaceLinearMap<CTensor> &A, size_t n);

RTensor eigs(const InPlaceLinearMap<RTensor> &f, size_t dim, EigType eig_type,
             size_t neig, RTensor *eigenvectors = nullptr,
             bool *converged = nullptr);

CTensor eigs(const InPlaceLinearMap<CTensor> &f, size_t dim, EigType eig_type,
             size_t neig, CTensor *eigenvectors = nullptr,
             bool *converged = nullptr);

CTensor eigs_gen(const InPlaceLinearMap<RTensor> &f, size_t dim,
                 EigType eig_type, size_t neig, CTensor *eigenvectors = nullptr,
                 bool *converged = nullptr);

CTensor eigs_gen(const InPlaceLinearMap<CTensor> &f, size_t dim,
                 EigType eig_type, size_t neig, CTensor *eigenvectors = nullptr,
                 bool *converged = nullptr);

RTensor eigs_small(const RTensor &A, EigType eig_type, size_t neig,
                   RTensor *eigenvectors, bool *converged);

RTensor eigs_small(const CTensor &A, EigType eig_type, size_t neig,
                   CTensor *eigenvectors, bool *converged);

CTensor eigs_gen_small(const RTensor &A, EigType eig_type, size_t neig,
                       CTensor *eigenvectors, bool *converged);

CTensor eigs_gen_small(const CTensor &A, EigType eig_type, size_t neig,
                       CTensor *eigenvectors, bool *converged);

}  // namespace arpack
#endif

#ifdef TENSOR_USE_PRIMME
namespace primme {

RTensor eigs(const InPlaceLinearMap<RTensor> &f, size_t dim, EigType eig_type,
             size_t neig, RTensor *eigenvectors = nullptr,
             bool *converged = nullptr);

CTensor eigs(const InPlaceLinearMap<CTensor> &f, size_t dim, EigType eig_type,
             size_t neig, CTensor *eigenvectors = nullptr,
             bool *converged = nullptr);

CTensor eigs_gen(const InPlaceLinearMap<CTensor> &f, size_t dim,
                 EigType eig_type, size_t neig, CTensor *eigenvectors = nullptr,
                 bool *converged = nullptr);

}  // namespace primme
#endif

#if defined(TENSOR_USE_ARPACK) || defined(TENSOR_USE_PRIMME)
enum EigsDriver {
  ArpackDriver = 0,
  PrimmeDriver = 1
};

#ifdef TENSOR_USE_ARPACK
#ifdef TENSOR_USE_PRIMME
EigsDriver get_default_eigs_driver();
void set_default_eigs_driver(EigsDriver driver);
#else
constexpr inline EigsDriver get_default_eigs_driver() {
  return ArpackDriver;
}
constexpr inline void set_default_eigs_driver(EigsDriver driver) {
  tensor_assert(driver == ArpackDriver);
}
#endif
#else
constexpr inline EigsDriver get_default_eigs_driver() {
  return PrimmeDriver;
}
constexpr inline void set_default_eigs_driver(EigsDriver driver) {
  tensor_assert(driver == PrimmeDriver);
}
#endif

/*------------ Complex Hermitian problems ----------- */

/**Find out a few eigenvalues and eigenvectors of a complex matrix. 'eigenvectors' is
     used to output the eigenvectors, but it can also contain a good estimate of
     them. 'converged' is true when the algorithm finished properly. */
CTensor eigs(const CTensor &A, EigType eig_type, size_t neig,
             CTensor *eigenvectors = nullptr, bool *converged = nullptr);

/**Find out a few eigenvalues and eigenvectors of a complex sparse matrix.
     'eigenvectors' is used to output the eigenvectors, but it can also contain a
     good estimate of them. 'converged' is true when the algorithm finished
     properly. */
CTensor eigs(const CSparse &A, EigType eig_type, size_t neig,
             CTensor *eigenvectors = nullptr, bool *converged = nullptr);

/**Find out a few eigenvalues and eigenvectors of a nonsymmetric real sparse
     matrix. 'f' is a function that takes in a Tensor and returns also a Tensor
     of the same class and dimension. Because we do not know the dimensions of
     'f', this has to be provided in 'dim'. 'eigenvectors' is used to output the
     eigenvectors, but it can also contain a good estimate of them. 'converged'
     is true when the algorithm finished properly. */
CTensor eigs(const LinearMap<CTensor> &A, size_t dim, EigType eig_type,
             size_t neig, CTensor *eigenvectors = nullptr,
             bool *converged = nullptr);

/**Find out a few eigenvalues and eigenvectors of a complex sparse matrix. 'f'
     is a function that takes in a Tensor and returns also a Tensor of the same
     class and dimension. Because we do not know the dimensions of 'f', this has
     to be provided in 'dim'. 'eigenvectors' is used to output the eigenvectors, but
     it can also contain a good estimate of them. 'converged' is true when the
     algorithm finished properly. */
CTensor eigs(const InPlaceLinearMap<CTensor> &f, size_t dim, EigType eig_type,
             size_t neig, CTensor *eigenvectors = nullptr,
             bool *converged = nullptr);

/*------------ Real symmetric problems ----------- */

/**Find out a few eigenvalues and eigenvectors of a real symmetric matrix.
     'eigenvectors' is used to output the eigenvectors, but it can also contain a
     good estimate of them. 'converged' is true when the algorithm finished
     properly. */
RTensor eigs(const RTensor &A, EigType eig_type, size_t neig,
             RTensor *eigenvectors = nullptr, bool *converged = nullptr);

/**Find out a few eigenvalues and eigenvectors of a symmetric real sparse
     matrix. 'eigenvectors' is used to output the eigenvectors, but it can also
     contain a good estimate of them. 'converged' is true when the algorithm
     finished properly. */
RTensor eigs(const RSparse &A, EigType eig_type, size_t neig,
             RTensor *eigenvectors = nullptr, bool *converged = nullptr);

/**Find out a few eigenvalues and eigenvectors of a nonsymmetric real sparse
     matrix. 'f' is a function that takes in a Tensor and returns also a Tensor
     of the same class and dimension. Because we do not know the dimensions of
     'f', this has to be provided in 'dim'. 'eigenvectors' is used to output the
     eigenvectors, but it can also contain a good estimate of them. 'converged'
     is true when the algorithm finished properly. */
RTensor eigs(const LinearMap<RTensor> &f, size_t dim, EigType eig_type,
             size_t neig, RTensor *eigenvectors = nullptr,
             bool *converged = nullptr);

/**Find out a few eigenvalues and eigenvectors of a symmetric real sparse
     matrix. 'f' is a function that takes in a Tensor and stores in the output
     argument a tensor of the same class and dimension. Because we do not know
     the dimensions of 'f', this has to be provided in 'dim'. 'eigenvectors' is used
     to output the eigenvectors, but it can also contain a good estimate of
     them. 'converged' is true when the algorithm finished properly. */
RTensor eigs(const InPlaceLinearMap<RTensor> &f, size_t dim, EigType eig_type,
             size_t neig, RTensor *eigenvectors = nullptr,
             bool *converged = nullptr);

/*------------ Real non-symmetric problems ----------- */

/**Find out a few eigenvalues and eigenvectors of a real nonsymmetric matrix.
     'eigenvectors' is used to output the eigenvectors, but it can also contain a
     good estimate of them. 'converged' is true when the algorithm finished
     properly. */
CTensor eigs_gen(const RTensor &A, EigType eig_type, size_t neig,
                 CTensor *eigenvectors = nullptr, bool *converged = nullptr);

/**Find out a few eigenvalues and eigenvectors of a nonsymmetric real sparse
     matrix. 'eigenvectors' is used to output the eigenvectors, but it can also
     contain a good estimate of them. 'converged' is true when the algorithm
     finished properly. */
CTensor eigs_gen(const RSparse &A, EigType eig_type, size_t neig,
                 CTensor *eigenvectors = nullptr, bool *converged = nullptr);

/**Find out a few eigenvalues and eigenvectors of a nonsymmetric real sparse
     matrix. 'f' is a function that takes in a Tensor and returns also a Tensor
     of the same class and dimension. Because we do not know the dimensions of
     'f', this has to be provided in 'dim'. 'eigenvectors' is used to output the
     eigenvectors, but it can also contain a good estimate of them. 'converged'
     is true when the algorithm finished properly. */
CTensor eigs_gen(const LinearMap<RTensor> &f, size_t dim, EigType eig_type,
                 size_t neig, CTensor *eigenvectors = nullptr,
                 bool *converged = nullptr);

/**Find out a few eigenvalues and eigenvectors of a nonsymmetric real sparse
     matrix. 'f' is a function that takes in a Tensor and stores in the output
     argument a tensor of the same class and dimension. Because we do not know
     the dimensions of 'f', this has to be provided in 'dim'. 'eigenvectors' is used
     to output the eigenvectors, but it can also contain a good estimate of
     them. 'converged' is true when the algorithm finished properly. */
CTensor eigs_gen(const InPlaceLinearMap<RTensor> &f, size_t dim,
                 EigType eig_type, size_t neig, CTensor *eigenvectors = nullptr,
                 bool *converged = nullptr);

/*------------ Complex non-symmetric problems ----------- */

/**Find out a few eigenvalues and eigenvectors of a complex nonsymmetric matrix.
     'eigenvectors' is used to output the eigenvectors, but it can also contain a
     good estimate of them. 'converged' is true when the algorithm finished
     properly. */
CTensor eigs_gen(const CTensor &A, EigType eig_type, size_t neig,
                 CTensor *eigenvectors = nullptr, bool *converged = nullptr);

/**Find out a few eigenvalues and eigenvectors of a nonsymmetric complex sparse
     matrix. 'eigenvectors' is used to output the eigenvectors, but it can also
     contain a good estimate of them. 'converged' is true when the algorithm
     finished properly. */
CTensor eigs_gen(const CSparse &A, EigType eig_type, size_t neig,
                 CTensor *eigenvectors = nullptr, bool *converged = nullptr);

/**Find out a few eigenvalues and eigenvectors of a nonsymmetric complex sparse
     matrix. 'f' is a function that takes in a Tensor and returns also a Tensor
     of the same class and dimension. Because we do not know the dimensions of
     'f', this has to be provided in 'dim'. 'eigenvectors' is used to output the
     eigenvectors, but it can also contain a good estimate of them. 'converged'
     is true when the algorithm finished properly. */
CTensor eigs_gen(const LinearMap<CTensor> &f, size_t dim, EigType eig_type,
                 size_t neig, CTensor *eigenvectors = nullptr,
                 bool *converged = nullptr);

/**Find out a few eigenvalues and eigenvectors of a nonsymmetric complex sparse
     matrix. 'f' is a function that takes in a Tensor and stores in the output
     argument a tensor of the same class and dimension. Because we do not know
     the dimensions of 'f', this has to be provided in 'dim'. 'eigenvectors' is used
     to output the eigenvectors, but it can also contain a good estimate of
     them. 'converged' is true when the algorithm finished properly. */
CTensor eigs_gen(const InPlaceLinearMap<CTensor> &f, size_t dim,
                 EigType eig_type, size_t neig, CTensor *eigenvectors = nullptr,
                 bool *converged = nullptr);

#endif

}  // namespace linalg

#endif
