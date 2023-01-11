#include "primme_core.h"

namespace linalg {

namespace primme {

using namespace tensor;

CTensor eigs(const InPlaceLinearMap<CTensor> &A, size_t n, EigType eig_type,
             size_t neig, CTensor *eigenvectors, bool *converged) {
  return do_primme<cdouble, double>(A, n, eig_type, neig, eigenvectors,
                                    converged);
}

CTensor eigs_gen(const InPlaceLinearMap<CTensor> &A, size_t n, EigType eig_type,
                 size_t neig, CTensor *eigenvectors, bool *converged) {
  return do_primme<cdouble, cdouble>(A, n, eig_type, neig, eigenvectors,
                                     converged);
}

}  // namespace primme

}  // namespace linalg
