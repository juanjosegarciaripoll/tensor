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

#ifndef TENSOR_ARPACK_H
#define TENSOR_ARPACK_H

#include <array>
#include <tensor/tensor_blas.h>
#include <tensor/linalg.h>

namespace linalg {

namespace arpack {

template <typename elt_t, bool symmetric>
struct eigenvalue_type {
  typedef elt_t type;
};
template <>
struct eigenvalue_type<double, true> {
  typedef double type;
};
template <>
struct eigenvalue_type<double, false> {
  typedef tensor::cdouble type;
};

}  // namespace arpack

/*!\addtogroup Linalg */
/*!@{*/

/**Finder of a few eigenvalues of eigenvectors via Arnoldi method.*/
template <typename scalar_t, bool is_symmetric = true>
class Arpack {
 public:
  using elt_t = scalar_t;
  using integer = blas::integer;
  using Tensor = tensor::Tensor<elt_t>;
  using eigenvalue_t =
      typename arpack::eigenvalue_type<elt_t, is_symmetric>::type;
  using eigenvector_t = tensor::Tensor<eigenvalue_t>;
  using eigenvalues_t = tensor::Tensor<eigenvalue_t>;
  static constexpr bool symmetric = is_symmetric;

  enum Status {
    Uninitialized = 0,
    Initialized = 1,
    Running = 2,
    Finished = 3,
    Error = 4,
    TooManyIterations = 5,
    NoConvergence = 6,
  };

  Arpack(size_t n, enum EigType t, size_t neig);
  void set_random_start_vector();
  void set_start_vector(const elt_t *v);
  void set_tolerance(double tol);
  void set_maxiter(size_t maxiter);
  enum Status update();
  elt_t *get_x_vector();
  elt_t *get_y_vector();
  const Tensor &get_x();
  Tensor &get_y();
  void set_y(const Tensor &y);
  eigenvalues_t get_data(eigenvector_t *vectors);
  std::string error_message() { return std::string(error); };
  enum Status get_status() { return status; };
  size_t get_vector_size() { return static_cast<size_t>(n); };

  static tensor::Indices sort_values(const tensor::CTensor &t,
                                     EigType selector);

 protected:
  enum Status status;
  enum EigType which_eig;

  integer n;          // Dimension of the eigenproblem.
  integer nev;        // Number of eigenvalues to be computed. 0 < nev < n-1.
  integer ncv;        // Number of Arnoldi vectors generated at each iteration.
  integer maxit;      // Maximum number of Arnoldi update iterations allowed.
  const char *which;  // Specify which of the Ritz values of OP to compute.
  double tol;         // Stopping criterion (relative accuracy of Ritz values).
  std::unique_ptr<elt_t[]> resid;  // Initial residual vector.

  // a.2) Internal variables.

  bool rvec;           // Indicates if eigenvectors/Schur vectors were
                       // requested (or only eigenvalues will be determined).
  char bmat;           // Indicates if the problem is a standard ('I') or
                       // generalized ('G") eigenproblem.
  char hwmny;          // Indicates if eigenvectors ('A') or Schur vectors ('P')
                       // were requested (not referenced if rvec = false).
  integer ido;         // Original ARPACK reverse communication flag.
  integer info;        // Original ARPACK error flag.
  integer mode;        // Indicates the type of the eigenproblem (regular,
                       // shift and invert, etc).
  integer lworkl;      // Dimension of array workl.
  integer lworkv;      // Dimension of array workv.
  integer lrwork;      // Dimension of array rwork.
  integer iparam[12];  // RVector that handles original ARPACK parameters.
  integer ipntr[15];   // RVector that handles original ARPACK pointers.
  std::unique_ptr<double[]> rwork;     // Original ARPACK internal vector.
  std::unique_ptr<elt_t[]> workl;      // Original ARPACK internal vector.
  std::unique_ptr<elt_t[]> workd;      // Original ARPACK internal vector.
  std::unique_ptr<elt_t[]> workv;      // Original ARPACK internal vector.
  std::unique_ptr<elt_t[]> V;          // Arnoldi basis / Schur vectors.
  std::array<Tensor, 3> work_vectors;  // Vectors constructed on top of workd

  // a.3) Pure output variables.

  integer nconv;  // Number of "converged" Ritz values.

  const char *error;
};

/*!@}*/

extern template class Arpack<double, true>;
extern template class Arpack<double, false>;
extern template class Arpack<tensor::cdouble, true>;

#ifdef DOXYGEN_ONLY
/** Arpack solver for vectors of type RTensor and real symmetric matrices. */
struct RArpack : public Arpack<double, true> {
}
/** Arpack solver for vectors of type CTensor and complex matrices. */
struct CArpack : public Arpack<tensor::cdouble, true> {
}
#else
typedef Arpack<double, true> RArpack;
typedef Arpack<tensor::cdouble, true> CArpack;
#endif

}  // namespace linalg

#endif  // TENSOR_ARPACK_H
