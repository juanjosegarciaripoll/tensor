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
#ifndef TENSOR_ARPACK_Z_H
#define TENSOR_ARPACK_Z_H

#include <tensor/tensor_blas.h>
#include <tensor/linalg.h>

namespace linalg {

/*!\addtogroup Linalg */
/*!@{*/

/**Finder of a few eigenvalues of eigenvectors via Arnoldi method.*/
class CArpack {
 public:
  typedef tensor::cdouble elt_t;
  typedef blas::integer integer;

  enum Status {
    Uninitialized = 0,
    Initialized = 1,
    Running = 2,
    Finished = 3,
    Error = 4,
    TooManyIterations = 5,
    NoConvergence = 6,
  };

  CArpack(size_t n, enum EigType t, size_t neig);
  ~CArpack();
  void set_random_start_vector();
  void set_start_vector(const elt_t *v);
  void set_tolerance(double tol);
  void set_maxiter(size_t maxiter);
  enum Status update();
  elt_t *get_x_vector();
  elt_t *get_y_vector();
  const tensor::CTensor get_x();
  tensor::CTensor get_y();
  void set_y(const tensor::CTensor &y);
  tensor::CTensor get_data(tensor::CTensor *vectors);
  tensor::CTensor get_data(tensor::cdouble *z);
  std::string error_message() { return std::string(error); };
  enum Status get_status() { return status; };
  size_t get_vector_size() { return n; };

  void prepare();
  void clear();

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
  elt_t sigma;        // Shift (for nonsymmetric problems).
  elt_t *resid;       // Initial residual vector.

  // a.2) Internal variables.

  bool symmetric;      // Symmetric matrix, or not
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
  integer iparam[12];  // CVector that handles original ARPACK parameters.
  integer ipntr[15];   // CVector that handles original ARPACK pointers.
  double *rwork;       // Original ARPACK internal vector.
  elt_t *workl;        // Original ARPACK internal vector.
  elt_t *workd;        // Original ARPACK internal vector.
  elt_t *workv;        // Original ARPACK internal vector.
  elt_t *V;            // Arnoldi basis / Schur vectors.

  // a.3) Pure output variables.

  integer nconv;  // Number of "converged" Ritz values.

  const char *error;
};

/*!@}*/

}  // namespace linalg

#endif /* !TENSOR_ARPACK_Z_H */
