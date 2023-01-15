#pragma once
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
#ifndef TENSOR_SPARSE_TYPES_H

#include <vector>
#include <tuple>
#include <tensor/tensor.h>

namespace tensor {

template <typename elt_t>
struct SparseTriplet {
  index_t row{0}, col{0};
  elt_t value{0};
  bool operator<(const SparseTriplet &other) const {
    if (row < other.row) return true;
    if (row == other.row) return col < other.col;
    return false;
  }
  SparseTriplet(index_t a_row, index_t a_column, elt_t a_value)
      : row{a_row}, col{a_column}, value{a_value} {}
};

inline std::tuple<index_t, index_t> guess_diagonal_matrix_dimensions(
    index_t max_diagonal_size, const Indices &diagonals) {
  index_t rows = max_diagonal_size, columns = max_diagonal_size;
  for (auto diagonal : diagonals) {
    if (diagonal > 0)
      columns = std::max(columns, max_diagonal_size + diagonal);
    else
      rows = std::max(rows, max_diagonal_size - diagonal);
  }
  return {rows, columns};
}

/**A sparse matrix. A sparse matrix is a compact representation of
     two-dimensional tensors that have a lot of zero elements. Our
     implementation behaves much like Matlab's sparse matrices, in the sense
     that one can build them up from 2D tensors, preallocate them, perform
     matrix multiplication with mmult(), etc, etc.

     \ingroup Tensors
  */
template <typename elt>
class CSRMatrix {
 public:
  using elt_t = elt;
  using tensor = Tensor<elt>;
  using triplet_t = SparseTriplet<elt>;
  using coordinates_t = std::vector<triplet_t>;

  /**Build an empty matrix.*/
  CSRMatrix();
  /**Create a matrix with all elements set to zero.*/
  CSRMatrix(index_t rows, index_t cols, index_t nonzero = 0);
  /**Create a matrix from a collection of coordinates and values.*/
  CSRMatrix(coordinates_t coordinates, index_t rows = 0, index_t columns = 0);
  /**Create a sparse matrix from the coordinates and values. */
  CSRMatrix(const Indices &row_indices, const Indices &column_indices,
            const Tensor<elt_t> &data, index_t rows = 0, index_t columns = 0);
  /* Create a sparse matrix from its internal representation. */
  CSRMatrix(Indices dims, Indices row_start, Indices column,
            Tensor<elt_t> data);
  /**Convert a tensor to sparse form.*/
  explicit CSRMatrix(const Tensor<elt_t> &tensor);
  /**Construct a sparse matrix from nested initializer lists.*/
  CSRMatrix(const typename detail::nested_initializer_list<2, elt_t>::type &l) : CSRMatrix(tensor(l)) {}
  /**Copy constructor.*/
  CSRMatrix(const CSRMatrix<elt_t> &s) = default;
  /**Move constructor.*/
  CSRMatrix(CSRMatrix<elt_t> &&s) = default;
  /**Assignment operator.*/
  CSRMatrix &operator=(const CSRMatrix<elt_t> &s) = default;
  /**Move assignment operator.*/
  CSRMatrix &operator=(CSRMatrix<elt_t> &&s) = default;
  /**Implicit conversion from other sparse types.*/
  // NOLINTNEXTLINE(*-explicit-constructor)
  template <typename e2>
  // cppcheck-suppress noExplicitConstructor
  CSRMatrix(const CSRMatrix<e2> &other)
      : dims_(other.priv_dims()),
        row_start_(other.priv_row_start()),
        column_(other.priv_column()),
        data_(other.priv_data()) {}
  ~CSRMatrix() = default;

  /**Return an element of the sparse matrix.*/
  elt_t operator()(index_t row, index_t col) const;

  /**Return CSRMatrix matrix dimensions.*/
  const Dimensions &dimensions() const { return dims_; }
  /**Length of a given CSRMatrix matrix index.*/
  index_t dimension(int which) const;
  /**Number of rows.*/
  index_t rows() const { return dims_[0]; }
  /**Number of columns*/
  index_t columns() const { return dims_[1]; }
  /**Number of nonzero elements.*/
  index_t length() const {
    index_t r = rows();
    return r ? row_start_[r] : 0;
  }

  /**Empty matrix?*/
  bool is_empty() const { return (rows() == 0) || (columns() == 0); }

  /**Identity matrix in sparse form.*/
  static CSRMatrix<elt_t> eye(index_t rows, index_t columns);
  /**Identity matrix in sparse form.*/
  static CSRMatrix<elt_t> eye(index_t rows) { return eye(rows, rows); }
  /**Return a random sparse matrix.*/
  static CSRMatrix<elt_t> random(index_t rows, index_t columns,
                                 double density = 0.2);

  template <typename t>
  friend const Tensor<t> full(const CSRMatrix<t> &s);

  static CSRMatrix diag(const RTensor &values, Indices which, index_t rows = 0,
                        index_t columns = 0) {
    coordinates_t triplets;
    triplets.reserve(values.size());
    tensor_assert(values.rank() == 2);
    tensor_assert(values.rows() == which.ssize());
    if (rows == 0 && columns == 0) {
      auto dims = guess_diagonal_matrix_dimensions(values.columns(), which);
      rows = std::get<0>(dims);
      columns = std::get<1>(dims);
    }
    for (index_t row = 0; row < values.rows(); ++row) {
      index_t diagonal = which[row];
      for (index_t column = 0; column < values.columns(); ++column) {
        index_t actual_row = diagonal > 0 ? column : column - diagonal;
        index_t actual_column = diagonal > 0 ? column + diagonal : column;
        if (actual_row >= rows || actual_column >= columns) break;
        triplets.emplace_back(actual_row, actual_column, values(row, column));
      }
    }
    return CSRMatrix(triplets, rows, columns);
  }

  static CSRMatrix diag(const RTensor &values, index_t which, index_t rows = 0,
                        index_t columns = 0) {
    tensor_assert(values.rank() == 1);
    return diag(reshape(values, 1, values.ssize()), Indices{which}, rows,
                columns);
  }

  const Dimensions &priv_dims() const { return dims_; }
  const Indices &priv_row_start() const { return row_start_; }
  const Indices &priv_column() const { return column_; }
  const Tensor<elt> &priv_data() const { return data_; }

 private:
  /** The dimensions (rows and columns) of the sparse matrix. */
  Dimensions dims_;
  /** Gives for each row of the matrix at which index the column_/data_ entries start. */
  Indices row_start_;
  /** Gives for each data_ entry the column in the matrix. */
  Indices column_;
  /** The single data entries. */
  Tensor<elt_t> data_;

  static coordinates_t make_sparse_triplets(const Indices &rows,
                                            const Indices &cols,
                                            const Tensor<elt_t> &data);

  static CSRMatrix make_sparse(coordinates_t sorted_data, index nrows,
                               index ncols);
};

template <typename elt_t>
using Sparse = CSRMatrix<elt_t>;

extern template class CSRMatrix<double>;
extern template class CSRMatrix<cdouble>;

using RSparse = Sparse<double>;
using CSparse = Sparse<cdouble>;

}  // namespace tensor

#endif  // TENSOR_SPARSE_TYPES
