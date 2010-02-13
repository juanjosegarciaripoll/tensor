// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <tensor/tensor.h>
#include <tensor/clapack.h>
#include <tensor/linalg.h>

namespace linalg {

  using namespace lapack;

  /*!\defgroup Linalg Linear algebra
   */

  bool accurate_svd = 0;

  /**Singular value decomposition of a real matrix.

     The singular value decomposition of a matrix A, consists in finding two
     unitary matrices U and V, and diagonal one S with nonnegative elements, such
     that \f$A = U S V\f$. The svd() routine computes the diagonal elements of
     the matrix S and puts them in a 1D tensor, which is the output of the
     routine.  Optionally, the matrices U and V are also computed, and stored in
     the variables pointed to by U and VT.

     Unless otherwise specified, if the matrix A has \c MxN elements, then U is
     \c MxM, V is \c NxN and the vector S will have \c min(M,N) elements. However
     if flag \c economic is different from zero, then we get smaller matrices,
     U being \c MxR, V being \c RxN and S will have \c R=min(M,N) elements.
     
     \ingroup Linalg
  */
  RTensor
  svd(RTensor A, RTensor *U, RTensor *VT, bool economic)
  {
    /*
      if (accurate_svd) {
      return block_svd(A, U, VT, economic);
      }
    */

    assert(A.rows() > 0);
    assert(A.columns() > 0);
    assert(A.rank() == 2);
    
    integer m = A.rows();
    integer n = A.columns();
    integer k = std::min(m, n);
    integer lwork, ldu, lda, ldv, info;
    RTensor output(k);
    double *work, *u, *v;
    double *a = tensor_pointer(A), *s = tensor_pointer(output), foo;
    char jobv[1], jobu[1];

    if (U) {
      *U = RTensor(m, economic? k : m);
      u = tensor_pointer(*U);
      jobu[0] = economic? 'S' : 'A';
      ldu = m;
    } else {
      jobu[0] = 'N';
      u = &foo;
      ldu = 1;
    }
    if (VT) {
      (*VT) = RTensor(economic? k : n, n);
      v = tensor_pointer(*VT);
      jobv[0] = economic? 'S' : 'A';
      ldv = economic? k : n;
    } else {
      jobv[0] = 'N';
      v = &foo;
      ldv = 1;
    }
    lda = m;
    lwork = -1;
    work = &foo;
    F77NAME(dgesvd)(jobu, jobv, &m, &n, a, &lda, s, u, &ldu, v, &ldv,
		    work, &lwork, &info);
    lwork = (int)work[0];
    work = new double[lwork];
    F77NAME(dgesvd)(jobu, jobv, &m, &n, a, &lda, s, u, &ldu, v, &ldv,
		    work, &lwork, &info);
    delete[] work;
    return output;
  }

} // namespace linalg
