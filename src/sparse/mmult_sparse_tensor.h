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

#ifndef TENSOR_MMULT_SPARSE_TENSOR_H
#define TENSOR_MMULT_SPARSE_TENSOR_H

//////////////////////////////////////////////////////////////////////
// RAW ROUTINES FOR THE SPARSE-TENSOR PRODUCT
//

template<typename elt_t>
static void
mult_sp_t(elt_t *dest,
	  const index *row_start, const index *column, const elt_t *matrix,
	  const elt_t *vector,
	  index i_len, index j_len, index k_len, index l_len)
{
    if (k_len == 1) {
#if 0
	// dest(i,l) = matrix(i,j) vector(j,l)
	for (; l_len; l_len--, vector+=j_len) {
	    for (index i = 0; i < i_len; i++) {
		elt_t accum = *dest;
		for (index j = row_start[i]; j < row_start[i+1]; j++) {
		    accum += matrix[j] * vector[column[j]];
		}
		*(dest++) = accum;
	    }
	}
#else
	for (; l_len; l_len--, vector+=j_len) {
	    const elt_t *m = matrix;
	    const index *c = column;
	    for (index i = 0; i < i_len; i++) {
		elt_t accum = *dest;
		for (index j = row_start[i+1] - row_start[i]; j; j--) {
		    accum += *(m++) * vector[*(c++)];
		}
		*(dest++) = accum;
	    }
	}
#endif
    } else {
	// dest(i,k,l) = matrix(i,j) vector(k,j,l)
	for (index l = 0; l < l_len; l++) {
	    const elt_t *v = vector + l*(k_len*j_len);
	    for (index k = 0; k < k_len; k++, v++) {
		for (index i = 0; i < i_len; i++) {
		    elt_t accum = *dest;
		    for (index j = row_start[i]; j < row_start[i+1]; j++) {
			accum += matrix[j] * v[column[j] * k_len];
		    }
		    *(dest++) = accum;
		}
	    }
	}
    }
}

//////////////////////////////////////////////////////////////////////
// HIGHER LEVEL INTERFACE
//

template<typename elt_t>
static inline const Tensor<elt_t>
do_mmult(const Sparse<elt_t> &m1, const Tensor<elt_t> &m2)
{
    Indices dims(m2.rank());
    index l_len = 1;
    for (index k = 1, N = m2.rank(); k < N; k++) {
	dims.at(k) = m2.dimension(k);
	l_len *= dims[k];
    }
    index j_len = m2.dimension(0);
    index i_len = dims.at(0) = m1.rows();

    if (j_len != m1.columns()) {
	std::cerr <<
	  "In mmult(S,T), the first index of tensor T does not match the number of\n"
	  "columns in sparse matrix S.";
	abort();
    }

    Tensor<elt_t> output = Tensor<elt_t>::zeros(dims);

    mult_sp_t<elt_t>(output.begin(),
                     m1.priv_row_start().begin(), m1.priv_column().begin(),
                     m1.priv_data().begin(),
                     m2.begin(),
                     i_len, j_len, 1, l_len);

    return output;
}

#endif /* !TENSOR_MMULT_SPARSE_TENSOR_H */
