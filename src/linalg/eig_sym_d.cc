// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <tensor/tensor.h>
#include <tensor/clapack.h>
#include <tensor/linalg.h>

namespace linalg {

  using namespace lapack;

  /**Eigenvalue decomposition of a real matrix.
     Given a square matrix A, we find a diagonal matrix D and a set of vectors R
     or L such that
     A V = V D
     and
     transpose(V) A = D transpose(V)

     The matrix A must be symmetric (transpose(A)==A).

     By default, only the diagonal elements of D are computed. However, also the
     matrix V can be computed if a pointer to the associated variable is
     supplied.

     \ingroup Linalg
  */
  RTensor
  eig_sym(const RTensor &A, RTensor *V)
  {
    if (accurate_svd)
      return block_eig_sym(A, V);

    integer n = A.rows();
    if ((size_t)n != A.columns()) {
      std::cerr << "Routine eig_sym() can only compute eigenvalues of square matrices, and you\n"
                << "have passed a matrix that is " << A.rows() << " by " << A.columns();
      abort();
    }

    RTensor aux(A);
    double *a = aux.pointer();
    integer lda = n, lwork, info[1];
    const char *jobz = (V == 0)? "N" : "V";
    const char *uplo = "U";
    RTensor output(n);
    double *w = tensor_pointer(output);

    lwork = -1;
    double work0[1];
    F77NAME(dsyev)(jobz, uplo, &n, a, &lda, w, work0, &lwork, info);
    lwork = (int)work0[0];

    double *work = new_atomic double[lwork];
    F77NAME(dsyev)(jobz, uplo, &n, a, &lda, w, work, &lwork, info);
    delete[] work;

    Indices ndx = sort_indices(output, false);
    output = output(Range(ndx));
    if (V) *V = aux(Range(), Range(ndx));
    return output;
  }

} // namespace linalg
