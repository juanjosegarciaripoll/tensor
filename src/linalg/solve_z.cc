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

#include <memory>
#include <tensor/tensor.h>
#include <tensor/tensor_lapack.h>
#include <tensor/linalg.h>

namespace linalg {

using namespace lapack;

/**Solve a complex linear system of equations by Gauss-Seidel method.

     \ingroup Linalg
  */
CTensor solve(const CTensor &A, const CTensor &B) {
  blas::integer n = tensor_rows(A);
  blas::integer lda = n;
  blas::integer ldb = tensor_rows(B);

  // Currently, we only solve square systems
  if (n != blas::tensor_columns(A)) {
    std::cerr
        << "Routine solve() can only operate on square systems of equations, "
           "i.e\n"
        << "when the number of unknowns is equal to the number of equations.\n"
        << "However, you have passed a matrix of size " << A.columns() << " by "
        << A.rows();
    abort();
  }
  // The size of B has to be compatible with that of A
  if (n != ldb) {
    std::cerr << "In solve(A,B), the number of equations does not match the "
                 "number of right\n"
              << "hand members. That is, while matrix A has " << n
              << " columns, the vector\n"
              << "B has " << ldb << " elements.";
    abort();
  }

  // The matrix that we pass to LAPACK is overwritten with the solution X
  CTensor output(B);
  cdouble *b = tensor_pointer(output);

  // Since B may be a tensor, we compute how many effective
  // right-hand-sides (nrhs) there are.
  auto nrhs = blas::index_to_blas(B.ssize()) / ldb;

  // The matrix that we pass to LAPACK is modified
  CTensor aux(A);
  cdouble *a = tensor_pointer(aux);

  auto ipiv = std::make_unique<blas::integer[]>(static_cast<size_t>(n));
  blas::integer info{};
#ifdef TENSOR_USE_ACML
  zgesv(n, nrhs, a, lda, ipiv, b, ldb, &info);
#else
  F77NAME(zgesv)(&n, &nrhs, a, &lda, ipiv.get(), b, &ldb, &info);
#endif

  if (info) {
    std::cerr << "In solve()\n"
                 "The matrix of the system of equations is singular and thus "
                 "the problem cannot be\n"
                 "solved with the standard resolutor.\n";
    abort();
  }

  return output;
}

}  // namespace linalg
