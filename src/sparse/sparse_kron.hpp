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

namespace tensor {

//////////////////////////////////////////////////////////////////////
// KRONECKER PRODUCT OF MATRICES
//

template <typename elt_t>
static Sparse<elt_t> do_kron(const Sparse<elt_t> &s2, const Sparse<elt_t> &s1) {
  index rows1 = s1.rows();
  index cols1 = s1.columns();
  index rows2 = s2.rows();
  index cols2 = s2.columns();
  index number_nonzero = s1.length() * s2.length();
  index total_rows = rows1 * rows2;
  index total_cols = cols1 * cols2;

  if (number_nonzero == 0) return Sparse<elt_t>(total_rows, total_cols);

  Tensor<elt_t> output_data(Dimensions({number_nonzero}));
  Indices output_column(number_nonzero);
  Indices output_row_start(total_rows + 1);
  Indices output_dims(igen << total_rows << total_cols);

  typename Tensor<elt_t>::iterator out_data = output_data.begin();
  typename Indices::iterator out_column = output_column.begin();
  typename Indices::iterator out_begin = out_column;
  typename Indices::iterator out_row_start = output_row_start.begin();

  // C([i,j],[k,l]) = s1(i,k) s2(j,l)
  *(out_row_start++) = 0;
  for (index l = 0; l < rows2; l++) {
    for (index k = 0; k < rows1; k++) {
      for (index j = s2.priv_row_start()[l]; j < s2.priv_row_start()[l + 1];
           j++) {
        for (index i = s1.priv_row_start()[k]; i < s1.priv_row_start()[k + 1];
             i++) {
          *(out_data++) = s1.priv_data()[i] * s2.priv_data()[j];
          *(out_column++) = s1.priv_column()[i] + cols1 * s2.priv_column()[j];
        }
      }
      *(out_row_start++) = out_column - out_begin;
    }
  }
  return Sparse<elt_t>(output_dims, output_row_start, output_column,
                       output_data);
}

}  // namespace tensor
