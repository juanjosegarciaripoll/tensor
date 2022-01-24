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

#include <algorithm>
#include <tensor/arpack.h>
#include "saupp.h"
#include "seupp.h"
#include "caupp.h"
#include "ceupp.h"

using namespace tensor;
using namespace linalg;

template <typename elt_t>
const char *eigenvalue_selector(enum EigType _t);

template <>
inline const char *eigenvalue_selector<double>(enum EigType _t) {
  static const char *whichs[6] = {"LM", "SM", "LA", "SA", NULL, NULL};
  if (_t == LargestImaginary) {
    std::cerr
        << "Cannot use LargestImaginary eigenvalue selector with real problems."
        << std::endl;
    abort();
  }
  if (_t == SmallestImaginary) {
    std::cerr << "Cannot use SmallestImaginary eigenvalue selector with real "
                 "problems."
              << std::endl;
    abort();
  }
  return whichs[_t];
}

template <>
inline const char *eigenvalue_selector<tensor::cdouble>(enum EigType _t) {
  static const char *whichs[6] = {"LM", "SM", "LR", "SR", "LI", "SI"};
  return whichs[_t];
}

template <typename elt_t>
Arpack<elt_t>::Arpack(size_t _n, enum EigType _t, size_t _nev) {
  if (_t < 0 || _t > 6) {
    std::cerr << "Invalid argument EigType passed to Arpack constructor"
              << std::endl;
    abort();
  }

  // Select the type of problem
  which_eig = _t;
  which = eigenvalue_selector<elt_t>(_t);
  symmetric = sizeof(elt_t) == sizeof(double);

  // Default tolerance is machine precision
  tol = -1.0;

  // Select the problem size
  n = _n;
  nev = _nev;

  // There's a limit in the number of eigenvalues we can get
  if ((nev == 0) || (nev > (n - 1))) {
    std::cerr << "Error in Arpack<elt_t>::Arpack(): \n "
              << "You request NEV=" << nev
              << " eigenvalues, while only between 1 and " << (n - 1)
              << " eigenvalues can be computed.";
    abort();
  }

  // Tell the library we are just beginning
  ido = 0;

  // By default, random vector
  info = 0;
  resid = std::make_unique<elt_t[]>(n);

  // Reserve space for the lanczos basis in which the eigenvectors are
  // approximated.
  ncv = std::min<blas::integer>(std::max<blas::integer>(2 * nev, 20), n);
  V = std::make_unique<elt_t[]>(n * ncv);

  // Conservative estimate by Matlab

  maxit = std::max<blas::integer>(
      300, (int)(ceil(2.0 * n / std::max<blas::integer>(ncv, 1))));

  // Parameters for the algorithm: the (-1) in the index is to make it
  // look like FORTRAN.
  for (size_t i = 1; i < 12; i++) iparam[i - 1] = 0;
  iparam[1 - 1] = 1;      // Shift produced by user
  iparam[3 - 1] = maxit;  // Maximum number of iterations
  iparam[4 - 1] = 1;      // Block size to be used in the recurrence
  iparam[7 - 1] = 1;      // Standard eigenvalue problem

  // Standard eigenvalue problem, A * x = lambda * x
  bmat = 'I';

  // When computing eigenvectors, compute them all
  hwmny = 'A';

  // Work space for reverse communication
  //
  lworkl = symmetric ? ncv * (ncv + 8) : ncv * (3 * ncv + 5);
  lworkv = ncv * 3;
  workd = std::make_unique<elt_t[]>(n * 3);
  workl = std::make_unique<elt_t[]>(lworkl);
  workv = std::make_unique<elt_t[]>(lworkv);
  rwork = std::make_unique<double[]>(ncv);
  for (size_t i = 0; i < 15; i++) ipntr[i] = 0;

  // We have initialized this structure
  status = Initialized;
}

template <typename elt_t>
typename Arpack<elt_t>::Status Arpack<elt_t>::update() {
  if (status < Initialized) {
    error = "ARPACK: Cannot call update() with an uninitialized ARPACK object.";
    status = Error;
    return status;
  }
  if (status > Running) {
    error = "ARPACK: Cannot call update() after the algorithm is finished.";
    status = Error;
    return status;
  }

  gen_aupp(ido, bmat, n, which, nev, tol, resid.get(), ncv, V.get(), n, iparam,
           ipntr, workd.get(), workl.get(), lworkl, rwork.get(), info);

  if (ido == 99) {
    status = Error;
    if (info == 1) {
      error = "Maximum number of iterations reached";
      status = TooManyIterations;
    } else if (info > 0) {
      error = "Algorithm failed to converge";
      status = NoConvergence;
    } else if (info < 0) {
      switch (info) {
        case -9:
          error = "Starting vector is zero";
          break;
        default:
          error = "Internal error -- some parameter is wrong";
          break;
      }
      std::cerr << error << ' ' << info << '\n';
      abort();
    } else {
      status = Finished;
    }
  } else if (ido != 1 && ido != -1) {
    error = "Internal error -- ARPACK asks for B matrix";
    status = Error;
  } else {
    status = Running;
  }
  return status;
}

template <typename elt_t>
Tensor<elt_t> Arpack<elt_t>::get_data(tensor::Tensor<elt_t> *vectors) {
  if (vectors) {
    *vectors = Tensor::empty(n, nev);
  }
  return get_data(vectors ? vectors->begin() : NULL);
}

template <typename elt_t>
Tensor<elt_t> Arpack<elt_t>::get_data(elt_t *z) {
  // Do we want eigenvectors?
  bool rvec;
  int ldz;
  if (z) {
    rvec = 1;
    ldz = n;
  } else {
    rvec = 0;
    ldz = 1;
  }

  // Room for eigenvalues
  auto output = Tensor::empty(nev + 1);
  elt_t *d = output.begin();

  // Unused here
  elt_t sigma = number_zero<elt_t>();

  gen_eupp(rvec, hwmny, d, z, ldz, sigma, workv.get(), bmat, n, which, nev, tol,
           resid.get(), ncv, V.get(), n, iparam, ipntr, workd.get(),
           workl.get(), lworkl, rwork.get(), info);
  if (info != 0) {
    static const char *const messages[17] = {
        "Unknown error.",
        "N must be positive.",
        "NEV must be positive.",
        "NCV must satisfy NEV < NCV <= N.",
        "WHICH be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.",
        "BMAT must be one of 'I' or 'G'.",
        "Length of private work workl array is not sufficient.",
        "Error return from trid. eigenvalue calculation.",
        "Starting vector is zero.",
        "IPARAM[7] must be 1,2,3,4,5.",
        "IPARAM[7] = 1 and BMAT = 'G' are incompatible.",
        "NEV and WHICH = 'BE' are incompatible.",
        "DSAUPP did not find any eigenvalues to sufficient accuracy.",
        "HOWMNY must be one of 'A' or 'S' if rvec = true.",
        "HOWMNY = 'S' not yet implemented."};
    std::cerr << "Routine Arpack<elt_t>::get_data() failed" << std::endl
              << messages[(info < -16 || info > 0) ? 0 : (-info)] << std::endl
              << "N=" << n << std::endl
              << "NEV=" << n << std::endl
              << "WHICH=" << which << std::endl
              << "BMAT=" << bmat << std::endl
              << "LWORKL=" << workl << std::endl;
    for (int i = 0; i < 12; i++)
      std::cerr << "IPARAM[" << i << "]=" << iparam[i] << std::endl;
    for (int i = 8; i < 11; i++)
      std::cerr << "IPNTR[" << i << "]=" << ipntr[i] << std::endl;
    abort();
  }
  return output(range(0, nev - 1));
}

template <typename elt_t>
void Arpack<elt_t>::set_start_vector(const elt_t *v) {
  if (status >= Running) {
    std::cerr << "Arpack<elt_t>:: Cannot change start vector while running\n";
    abort();
  }
  info = 1;
  memcpy(resid.get(), v, n * sizeof(elt_t));
}

template <typename elt_t>
void Arpack<elt_t>::set_random_start_vector() {
  info = 0;
}

template <typename elt_t>
void Arpack<elt_t>::set_tolerance(double _tol) {
  if (status >= Running) {
    std::cerr << "Arpack<elt_t>:: Cannot change tolerance while running\n";
    abort();
  }
  tol = _tol;
}

template <typename elt_t>
void Arpack<elt_t>::set_maxiter(size_t new_maxiter) {
  if (status >= Running) {
    std::cerr
        << "Arpack<elt_t>:: Cannot change number of iterations while running\n";
    abort();
  }
  maxit = new_maxiter;
  iparam[2] = maxit;
}

template <typename elt_t>
elt_t *Arpack<elt_t>::get_x_vector() {
  if (status != Running) {
    std::cerr << "Arpack<elt_t>:: get_x_vector() invoked outside main loop";
    abort();
  }
  // IPNTR[1] has a FORTRAN index, which is one-based, instead of zero-based
  return &workd[ipntr[1 - 1] - 1];
}

template <typename elt_t>
elt_t *Arpack<elt_t>::get_y_vector() {
  if (status != Running) {
    std::cerr << "Arpack<elt_t>:: get_y_vector() invoked outside main loop";
    abort();
  }
  // IPNTR[2] has a FORTRAN index, which is one-based, instead of zero-based
  return &workd[ipntr[2 - 1] - 1];
}

template <typename elt_t>
const tensor::Tensor<elt_t> Arpack<elt_t>::get_x() {
  return Tensor(tensor::Vector<elt_t>(n, get_x_vector()));
}

template <typename elt_t>
tensor::Tensor<elt_t> Arpack<elt_t>::get_y() {
  return Tensor(tensor::Vector<elt_t>(n, get_y_vector()));
}

template <typename elt_t>
void Arpack<elt_t>::set_y(const tensor::Tensor<elt_t> &y) {
  memcpy(get_y_vector(), y.begin(), sizeof(elt_t) * n);
}

template <typename elt_t>
Indices Arpack<elt_t>::sort_values(const CTensor &values, EigType t) {
  RTensor aux;
  switch (t) {
    case LargestReal:
      aux = -tensor::real(values);
      break;
    case LargestMagnitude:
      aux = -abs(values);
      break;
    case SmallestMagnitude:
      aux = abs(values);
      break;
    case LargestImaginary:
      aux = -imag(values);
      break;
    case SmallestImaginary:
      aux = imag(values);
      break;
    case SmallestReal:
      aux = tensor::real(values);
      break;
  }
  return sort_indices(aux);
}

/// Local variables:
/// mode: c++
/// fill-column: 80
/// c-basic-offset: 4
