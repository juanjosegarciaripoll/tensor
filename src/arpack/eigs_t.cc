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
// MPS library
//
// (c) 2004 Juan Jose Garcia Ripoll
//
//----------------------------------------------------------------------
// ARPACK DRIVER FOR NONSYMMETRIC EIGENVALUE PROBLEMS
//

#include <sys/blas.h>
#include <utils.h>
#include <linalg.h>
#include <arpack.h>

#undef COMPLEX

Tensor
eigs(const Tensor &A, int eig_type, size_t neig, Tensor *eigenvectors,
     const Tensor::elt_t *initial_guess)
{
    Arpack::EigType t = (Arpack::EigType)eig_type;
    size_t n = A.columns();

    if ((A.rank() != 2) || (A.rows() != n)) {
	std::cerr << "In eigs(): Can only compute eigenvalues of square matrices.";
	myabort();
    }

    if (neig > n || neig == 0) {
	std::cerr << "In eigs(): Can only compute up to " << n << " eigenvalues\n"
		  << "in a matrix that has " << n << " times " << n << " elements.";
	myabort();
    }

    if (n <= 4) {
	/* For small sizes, the ARPACK solver produces wrong results!
	 * In any case, for these sizes it is more efficient to do the solving
	 * using the full routine.
	 */
	CTensor vectors;
	CTensor values = eig(A, NULL, eigenvectors? &vectors : 0);
	UIVector ndx = Arpack::sort_values(values, t)(Range(0, neig-1));
#ifdef COMPLEX
	if (eigenvectors) {
	    *eigenvectors = vectors(Range(), Range(ndx));
	}
	return CTensor(values(Range(ndx)));
#else
	if (eigenvectors) {
	    *eigenvectors = re_part(vectors(Range(), Range(ndx)));
	}
	return re_part(values(Range(ndx)));
#endif
    }

    Tensor output;
    {
    Arpack data(n, t, neig);

    if (initial_guess)
	data.set_start_vector(initial_guess);

    while (data.update() < Arpack::Finished) {
	gemv('N', n, n, Tensor::elt_one(), A.pointer(), n, data.get_x_vector(), 1,
	     Tensor::elt_zero(), data.get_y_vector(), 1);
    }
    if (data.get_status() != Arpack::Finished) {
	std::cerr << data.error_message() << '\n';
	myabort();
    }
    output = data.get_data(eigenvectors);
    }
    return output;
}


/// Local variables:
/// mode: c++
/// fill-column: 80
/// c-basic-offset: 4
