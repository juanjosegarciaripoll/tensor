#include "primme_core.h"

namespace linalg {

namespace primme {

RTensor eigs(const InPlaceLinearMap<RTensor> &A, size_t n, EigType eig_type,
             size_t neig, RTensor *eigenvectors, bool *converged) {
  return do_primme<double, double>(A, n, eig_type, neig, eigenvectors,
                                   converged);
}

}  // namespace primme

}  // namespace linalg
