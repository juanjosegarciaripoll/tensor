// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <tensor/tensor.h>
#include <tensor/clapack.h>
#include <tensor/linalg.h>

namespace linalg {

  using namespace lapack;

  /**Singular value decomposition of a complex matrix.
     
     \ingroup Linalg
  */
  RTensor
  svd(CTensor A, CTensor *U, CTensor *VT, bool economic)
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
    cdouble *work, *u, *v, *a = tensor_pointer(A), foo;
    double *rwork, *s = tensor_pointer(output);
    char jobv[1], jobu[1];
    
    if (U) {
      *U = CTensor(m, economic? k : m);
      u = tensor_pointer(*U);
      jobu[0] = economic? 'S' : 'A';
      ldu = m;
    } else {
      jobu[0] = 'N';
      u = &foo;
      ldu = 1;
    }
    if (VT) {
      (*VT) = CTensor(economic? k : n, n);
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
    rwork = (double *)&foo;
    F77NAME(zgesvd)(jobu, jobv, &m, &n, a, &m, s, u, &ldu, v, &ldv,
		    work, &lwork, rwork, &info);
    lwork = lapack::real(work[0]);
    work = new cdouble[lwork];
    rwork = new double[5 * k];
    F77NAME(zgesvd)(jobu, jobv, &m, &n, a, &m, s, u, &ldu, v, &ldv,
		    work, &lwork, rwork, &info);
    delete[] work;
    delete[] rwork;
    return output;
  }


} // namespace linalg
