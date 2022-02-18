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

//////////////////////////////////////////////////////////////////////
// CREATE A SET OF INDICES COVERING i0:i1:i2 (MATLAB NOTATION)
//

static Range range2(index i0, index i2, index i1) {
  index l = (i2 - i0) / i1 + 1;
  Indices output(l);
  for (int i = 0; i < l; i++, i0 += i1) {
    output.at(i) = i0;
  }
  return Range(output);
}
