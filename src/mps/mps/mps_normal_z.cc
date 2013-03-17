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

#include <mps/mps.h>

namespace mps {

  const CMPS normal_form(const CMPS &P, int sense)
  {
    CMPS output = canonical_form(P, sense);
    index i = (sense < 0) ? 0 : P.size()-1;
    const CTensor &t = output[i];
    double fac = P.is_periodic()? norm2(output) : norm2(t);
    output.at(i) = t / fac;
    return output;
   }

} // namespace mps
