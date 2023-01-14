#pragma once

#include <tensor/tensor.h>
#include <tensor/linalg.h>
#include <primme.h>

namespace linalg {

namespace primme {

using namespace tensor;

template <typename elt_t>
static void realMatrixMatvec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy,
                             int *blockSize, primme_params *primme,
                             int * /*ierr*/) {
  auto A = static_cast<const InPlaceLinearMap<Tensor<elt_t>> *>(primme->matrix);
  auto xvector = Tensor<elt_t>::from_pointer(
      Dimensions{static_cast<index_t>(*ldx), static_cast<index_t>(*blockSize)},
      static_cast<elt_t *>(x));
  auto yvector = Tensor<elt_t>::from_pointer(
      Dimensions{static_cast<index_t>(*ldy), static_cast<index_t>(*blockSize)},
      static_cast<elt_t *>(y));
  (*A)(xvector, yvector);
}

static inline int call_primme(double *evals, double *evecs, double *rnorms,
                              primme_params *primme) {
  return dprimme(evals, evecs, rnorms, primme);
}

static inline int call_primme(double *evals, cdouble *evecs, double *rnorms,
                              primme_params *primme) {
  return zprimme(evals, evecs, rnorms, primme);
}

static inline int call_primme(cdouble *evals, cdouble *evecs, double *rnorms,
                              primme_params *primme) {
  return zprimme_normal(evals, evecs, rnorms, primme);
}

template <typename elt_t, typename eigenvalue_t = elt_t>
static Tensor<eigenvalue_t> do_primme(const InPlaceLinearMap<Tensor<elt_t>> &A,
                                      size_t n, EigType eig_type, size_t neig,
                                      Tensor<elt_t> *eigenvectors,
                                      bool *converged) {
  primme_params primme;

  primme.n = static_cast<PRIMME_INT>(n);
  primme.numEvals = static_cast<PRIMME_INT>(neig);
  primme.eps = std::numeric_limits<double>::epsilon();  // options.epsilon();

  auto evals = Tensor<eigenvalue_t>::empty(primme.numEvals);
  auto evecs = Tensor<elt_t>::empty(primme.n, primme.numEvals);
  auto rnorms = Tensor<double>::empty(primme.numEvals);
  std::unique_ptr<double> shifts;

  switch (eig_type) {
    case LargestMagnitude:
      primme.target = primme_largest_abs;
      shifts = std::make_unique<double>(0.0);
      primme.targetShifts = shifts.get();
      break;
    case LargestAlgebraic:
      primme.target = primme_largest;
      break;
    case SmallestAlgebraic:
      primme.target = primme_smallest;
      break;
    default:
      std::cerr << "Unsupported eigenvalue type in primme() " << eig_type
                << '\n';
      std::abort();
  }

  primme_set_method(PRIMME_DYNAMIC, &primme);

  primme.matrixMatvec = realMatrixMatvec<elt_t>;
  primme.matrix = const_cast<void *>(static_cast<const void *>(&A));

  int ret = call_primme(evals.begin(), evecs.begin(), rnorms.begin(), &primme);
  primme_free(&primme);

  if (ret < 0) {
    std::cerr << "primme: return with nonzero exit status " << ret << '\n';
    *converged = false;
  } else {
    *converged = true;
  }

  if (eigenvectors) *eigenvectors = std::move(evecs);
  return evals;
}

}  // namespace primme

}  // namespace linalg
