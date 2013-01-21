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

using namespace tensor;
using namespace linalg;

ARPACK::ARPACK(size_t _n, enum EigType _t, size_t _nev)
{
#ifdef COMPLEX
    static const char *whichs[6] = {"LM", "SM", "LR", "SR", "LI", "SI"};
#else
    static const char *whichs[4] = {"LM", "SM", "LA", "SA"};
#endif

    // Select the type of problem
    which_eig = _t;
    which = whichs[which_eig];
#ifdef COMPLEX
    symmetric = 0;
#else
    symmetric = 1;
#endif

    // Default tolerance is machine precision
    tol = -1.0;

    // Select the problem size
    n = _n;
    nev = _nev;

    // There's a limit in the number of eigenvalues we can get
    if ((nev == 0) || (nev > (n-1))) {
	std::cerr << "Error in ARPACK::ARPACK(): \n "
		  << "You request NEV=" << nev << " eigenvalues, while only between 1 and "
		  << (n-1) << " eigenvalues can be computed.";
	abort();
    }

    // Tell the library we are just beginning
    ido = 0;

    // By default, random vector
    info = 0;
    resid = new ELT_T[n];

    // Reserve space for the lanczos basis in which the eigenvectors are
    // approximated.
    ncv = std::min(std::max(2 * nev, 20), n);
    V = new ELT_T[n * ncv];

    // Conservative estimate by Matlab
    maxit = std::max(300,(int)(ceil(2.0*n/std::max(ncv,1))));

    // Parameters for the algorithm: the (-1) in the index is to make it
    // look like FORTRAN.
    for (size_t i = 1; i < 12; i++)
	iparam[i-1] = 0;
    iparam[1-1] = 1;		// Shift produced by user
    iparam[3-1] = maxit;	// Maximum number of iterations
    iparam[4-1] = 1;		// Block size to be used in the recurrence
    iparam[7-1] = 1;		// Standard eigenvalue problem

    // Standard eigenvalue problem, A * x = lambda * x
    bmat = 'I';

    // When computing eigenvectors, compute them all
    hwmny = 'A';

    // Work space for reverse communication
    //
    lworkl = symmetric? ncv*(ncv + 8) : ncv*(3*ncv + 5);
    lworkv = ncv * 3;
    workd = new ELT_T[n * 3];
    workl = new ELT_T[lworkl];
    workv = new ELT_T[lworkv];
    rwork = new double[ncv];
    for (size_t i = 0; i < 15; i++)
	ipntr[i] = 0;

    // We have initialized this structure
    status = Initialized;
}

ARPACK::~ARPACK() {

    // Deleting working arrays
    delete[] workd; workd = 0;
    delete[] workl; workl = 0;
    delete[] workv; workv = 0;
    delete[] rwork; rwork = 0;
    delete[] V; V = 0;

    // Deleting input and output arrays
    delete[] resid; resid = 0;
}

ARPACK::Status ARPACK::update() {

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

#ifdef COMPLEX
    caupp(ido, bmat, n, which, nev, tol, resid, ncv, V, n, iparam, ipntr, workd,
	  workl, lworkl, rwork, info);
#else
    saupp(ido, bmat, n, which, nev, tol, resid, ncv, V, n, iparam, ipntr, workd,
	  workl, lworkl, info);
#endif

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
		case -9: error = "Starting vector is zero"; break;
		default: error = "Internal error -- some parameter is wrong"; break;
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

Tensor<ELT_T> ARPACK::get_data(Tensor<ELT_T> *vectors) {
    if (vectors) {
	*vectors = Tensor<ELT_T>(n, nev);
    }
    return get_data(vectors? vectors->begin() : NULL);
}

Tensor<ELT_T> ARPACK::get_data(Tensor<ELT_T>::elt_t *z) {
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
    Tensor<ELT_T> output(nev+1);
    ELT_T *d = output.begin();

    // Unused here
    ELT_T sigma = number_zero<ELT_T>();

#ifdef COMPLEX
    ceupp(rvec, hwmny, d, z, ldz, sigma, workv, bmat, n, which, nev, tol,
	  resid, ncv, V, n, iparam, ipntr, workd, workl, lworkl, rwork, info);
#else
    seupp(rvec, hwmny, d, z, ldz, sigma, bmat, n, which, nev, tol,
	  resid, ncv, V, n, iparam, ipntr, workd, workl, lworkl, info);
#endif
    if (info != 0) {
	std::cerr << "Routine ARPACK::get_data() failed";
	abort();
    }
    return output(range(0,nev-1));
}

void ARPACK::set_start_vector(const ELT_T *v) {
    if (status >= Running) {
	std::cerr << "ARPACK:: Cannot change start vector while running\n";
	abort();
    }
    info = 1;
    memcpy(resid, v, n * sizeof(ELT_T));
}

void ARPACK::set_random_start_vector() {
    info = 0;
}

void ARPACK::set_tolerance(double _tol) {
    if (status >= Running) {
	std::cerr << "ARPACK:: Cannot change tolerance while running\n";
	abort();
    }
    tol = _tol;
}

void ARPACK::set_maxiter(size_t new_maxiter) {
    if (status >= Running) {
	std::cerr << "ARPACK:: Cannot change number of iterations while running\n";
	abort();
    }
    maxit = new_maxiter;
    iparam[2] = maxit;
}

ELT_T *ARPACK::get_x_vector() {
    if (status != Running) {
	std::cerr << "ARPACK:: get_x_vector() invoked outside main loop";
	abort();
    }
    // IPNTR[1] has a FORTRAN index, which is one-based, instead of zero-based
    return &workd[ipntr[1-1]-1];
}

ELT_T *ARPACK::get_y_vector() {
    if (status != Running) {
	std::cerr << "ARPACK:: get_y_vector() invoked outside main loop";
	abort();
    }
    // IPNTR[2] has a FORTRAN index, which is one-based, instead of zero-based
    return &workd[ipntr[2-1]-1];
}

const Tensor<ELT_T> ARPACK::get_x()
{
    return Vector<ELT_T>(n, get_x_vector());
}

Tensor<ELT_T> ARPACK::get_y()
{
    return Vector<ELT_T>(n, get_y_vector());
}

void ARPACK::set_y(const Tensor<ELT_T> &y)
{
   memcpy(get_y_vector(), y.begin(), sizeof(Tensor<ELT_T>::elt_t)*n);
}

Indices
ARPACK::sort_values(const CTensor &values, ARPACK::EigType t)
{
    RTensor aux;
    switch (t) {
    case ARPACK::LargestReal:
	aux = -real(values);
	break;
    case ARPACK::LargestMagnitude:
	aux = -abs(values);
	break;
    case ARPACK::SmallestMagnitude:
	aux = abs(values);
	break;
#ifdef COMPLEX
    case ARPACK::LargestImag:
	aux = -imag(values);
	break;
    case ARPACK::SmallestImag:
	aux = imag(values);
	break;
#endif
    default:
    case ARPACK::SmallestReal:
	aux = real(values);
	break;
    }
    return sort_indices(aux);
}

/// Local variables:
/// mode: c++
/// fill-column: 80
/// c-basic-offset: 4
