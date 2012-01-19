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

#include <tensor/tensor.h>
#include <tensor/tensor_lapack.h>
#include <tensor/linalg.h>

namespace linalg {

  using namespace lapack;

  /**Solve a complex linear system of equations by Gauss-Seidel method.

     \ingroup Linalg
  */
  const CTensor
  solve(const CTensor &A, const CTensor &B) {
    integer n = A.rows();
    integer lda = n;
    integer ldb = B.dimension(0);
    integer nrhs;
    integer *ipiv = new int[n];
    integer info;

    // Currently, we only solve square systems
    if (n != (integer)A.columns()) {
      std::cerr << "Routine solve() can only operate on square systems of equations, i.e\n"
		<< "when the number of unknowns is equal to the number of equations.\n"
		<< "However, you have passed a matrix of size " << A.columns() << " by " << A.rows();
      abort();
    }
    // The size of B has to be compatible with that of A
    if (n != ldb) {
      std::cerr << "In solve(A,B), the number of equations does not match the number of right\n"
		<< "hand members. That is, while matrix A has " << n << " columns, the vector\n"
		<< "B has " << ldb << " elements.";
      abort();
    }

    // The matrix that we pass to LAPACK is overwritten with the solution X
    CTensor output(B);
    cdouble *b = tensor_pointer(output);

    // Since B may be a tensor, we compute how many effective
    // right-hand-sides (nrhs) there are.
    nrhs = B.size() / ldb;

    // The matrix that we pass to LAPACK is modified
    CTensor aux(A);
    cdouble *a = tensor_pointer(aux);

    F77NAME(zgesv)(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    delete[] ipiv;

    if (info) {
      std::cerr <<
	"In solve()\n"
	"The matrix of the system of equations is singular and thus the problem cannot be\n"
	"solved with the standard resolutor.\n";
      abort();
    }

    return output;
  }

}
