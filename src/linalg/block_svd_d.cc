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

#include "block_svd.hpp"

namespace linalg {

/**Singular value decomposition of a real matrix by blocks.

     The difference between this function and svd() is that this one performs
     the decomposition in blocks, preserving the symmetries of the original
     matrix.
     
     \ingroup Linalg
  */
RTensor block_svd(RTensor A, RTensor *pU, RTensor *pVT, bool economic) {
  return do_block_svd<RTensor>(A, pU, pVT, economic);
}

}  // namespace linalg
