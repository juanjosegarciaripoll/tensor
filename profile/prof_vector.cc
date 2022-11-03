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

#include <tensor/tensor.h>
#include <iostream>
#include <fstream>
#include <tuple>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <tensor/io.h>

#include "profile.h"

namespace benchmark {

using namespace tensor;

template <class T1, class T2>
void add(std::tuple<T1, T2> &args) {
  force(std::get<0>(args) + std::get<1>(args));
}

template <class T1, class T2>
void subtract(std::tuple<T1, T2> &args) {
  force(std::get<0>(args) - std::get<1>(args));
}

template <class T1, class T2>
void multiply(std::tuple<T1, T2> &args) {
  force(std::get<0>(args) * std::get<1>(args));
}

template <class T1, class T2>
void divide(std::tuple<T1, T2> &args) {
  force(std::get<0>(args) / std::get<1>(args));
}

template <class T1, class T2>
void add_inplace(std::tuple<T1, T2> &args) {
  force(std::get<0>(args) += std::get<1>(args));
}

template <class T1, class T2>
void subtract_inplace(std::tuple<T1, T2> &args) {
  force(std::get<0>(args) -= std::get<1>(args));
}

template <class T1, class T2>
void multiply_inplace(std::tuple<T1, T2> &args) {
  force(std::get<0>(args) *= std::get<1>(args));
}

template <class T1, class T2>
void divide_inplace(std::tuple<T1, T2> &args) {
  force(std::get<0>(args) /= std::get<1>(args));
}

template <class T>
void apply_sum(std::tuple<T> &args) {
  force_nonzero(sum(std::get<0>(args)));
}

template <class T>
void apply_exp(std::tuple<T> &args) {
  force(exp(std::get<0>(args)));
}

template <class T>
void apply_cos(std::tuple<T> &args) {
  force(cos(std::get<0>(args)));
}

template <class T>
void vector_const_indexed_read(std::tuple<T, typename T::elt_t> &args) {
  const T &v = std::get<0>(args);
  auto n = std::get<1>(args);
  auto x = n;
  for (tensor::index i = 0; i < static_cast<tensor::index>(v.size()); ++i) {
    x += v[i];
  }
  force_nonzero(x);
}

template <class T>
void vector_indexed_read(std::tuple<T, typename T::elt_t> &args) {
  T &v = std::get<0>(args);
  auto n = std::get<1>(args);
  auto x = n;
  for (tensor::index i = 0; i < static_cast<tensor::index>(v.size()); ++i) {
    x += v[i];
  }
  force_nonzero(x);
}

template <class T>
void vector_indexed_write(std::tuple<T, typename T::elt_t> &args) {
  T &v = std::get<0>(args);
  auto n = std::get<1>(args);
  for (tensor::index i = 0; i < static_cast<tensor::index>(v.size()); ++i) {
    v.at(i) = n;
  }
}

template <class T>
void copy_first_column(std::tuple<T, T> &args) {
  T &A = std::get<0>(args);
  T &B = std::get<1>(args);
  A.at(_, range(0)) = B(_, range(0));
}

template <class T>
void copy_first_row(std::tuple<T, T> &args) {
  T &A = std::get<0>(args);
  T &B = std::get<1>(args);
  A.at(range(0), _) = B(range(0), _);
}

template <class T>
void copy_first_column_index(std::tuple<T, T, Indices> &args) {
  T &A = std::get<0>(args);
  T &B = std::get<1>(args);
  Indices &ndx = std::get<2>(args);
  A.at(ndx, range(0)) = B(ndx, range(0));
}

template <class T>
void copy_first_row_index(std::tuple<T, T, Indices> &args) {
  T &A = std::get<0>(args);
  T &B = std::get<1>(args);
  Indices &ndx = std::get<2>(args);
  A.at(range(0), ndx) = B(range(0), ndx);
}

template <class T>
void extract_first_column(std::tuple<T, T> &args) {
  T &A = std::get<0>(args);
  T B = A(_, range(0));
  force(B);
}

template <class T>
void extract_first_row(std::tuple<T, T> &args) {
  T &A = std::get<0>(args);
  T B = A(range(0), _);
  force(B);
}

template <class T>
void fold_ij_jk(std::tuple<T, T> &args) {
  const T &A = std::get<0>(args);
  const T &B = std::get<1>(args);
  force(fold(A, -1, B, 0));
}

template <class T>
void fold_ij_kj(std::tuple<T, T> &args) {
  const T &A = std::get<0>(args);
  const T &B = std::get<1>(args);
  force(fold(A, -1, B, -1));
}

template <class T>
void fold_ji_kj(std::tuple<T, T> &args) {
  const T &A = std::get<0>(args);
  const T &B = std::get<1>(args);
  force(fold(A, 0, B, -1));
}

template <class T>
void fold_ji_jk(std::tuple<T, T> &args) {
  const T &A = std::get<0>(args);
  const T &B = std::get<1>(args);
  force(fold(A, 0, B, 0));
}

template <class T>
void mmult_N_N(std::tuple<T, T> &args) {
  const T &A = std::get<0>(args);
  const T &B = std::get<1>(args);
  force(mmult(A, B));
}

template <class T>
void mmult_T_N(std::tuple<T, T> &args) {
  const T &A = std::get<0>(args);
  const T &B = std::get<1>(args);
  force(mmult(transpose(A), B));
}

template <class T>
void mmult_N_T(std::tuple<T, T> &args) {
  const T &A = std::get<0>(args);
  const T &B = std::get<1>(args);
  force(mmult(A, transpose(B)));
}

template <class T>
void mmult_T_T(std::tuple<T, T> &args) {
  const T &A = std::get<0>(args);
  const T &B = std::get<1>(args);
  force(mmult(transpose(A), transpose(B)));
}

template <class T>
void warmup(size_t size) {
  for (int i = 0; i < 10; ++i) {
    std::unique_ptr<T> p{new T(Dimensions{size})};
    force(*p);
  }
}

template <class T>
std::tuple<T, typename T::elt_t> make_vector_and_number(size_t size) {
  auto number = static_cast<typename T::elt_t>(3.0);
  return typename std::tuple<T, typename T::elt_t>(
      T::random(static_cast<tensor::index>(size)), number);
}

template <class T>
std::tuple<T, T> make_two_vectors(size_t size) {
  return std::tuple<T, T>(T::random(static_cast<tensor::index>(size)),
                          T::random(static_cast<tensor::index>(size)) + 1.0);
}

template <class T>
std::tuple<T> make_vector(size_t size) {
  return typename std::tuple<T>(T::random(static_cast<tensor::index>(size)));
}

template <class T>
std::tuple<T, typename T::elt_t> make_vector_and_one(size_t size) {
  auto number = static_cast<typename T::elt_t>(1.0);
  return typename std::tuple<T, typename T::elt_t>(
      T::random(static_cast<tensor::index>(size)), number);
}

template <class T>
std::tuple<T, T> make_two_columns(size_t size) {
  size_t columns = 50;
  Dimensions d = {static_cast<tensor::index>(size),
                  static_cast<tensor::index>(columns)};
  return std::tuple<T, T>(T::random(d), T::random(d));
}

template <class T>
std::tuple<T, T, Indices> make_two_columns_and_index(size_t size) {
  auto aux = make_two_columns<T>(size);
  Indices ndx = iota(0, static_cast<tensor::index>(size) - 1, 2);
  return std::tuple<T, T, Indices>(std::get<0>(aux), std::get<1>(aux), ndx);
}

template <class T>
std::tuple<T, T> make_two_rows(size_t size) {
  size_t rows = 50;
  Dimensions d = {static_cast<tensor::index>(rows),
                  static_cast<tensor::index>(size)};
  return std::tuple<T, T>(T::random(d), T::random(d));
}

template <class T>
std::tuple<T, T, Indices> make_two_rows_and_index(size_t size) {
  auto aux = make_two_rows<T>(size);
  Indices ndx = iota(0, size - 1, 2);
  return std::tuple<T, T, Indices>(std::get<0>(aux), std::get<1>(aux), ndx);
}

template <class T>
std::tuple<T, T> make_two_matrices(size_t size) {
  Dimensions d = {static_cast<tensor::index>(size),
                  static_cast<tensor::index>(size)};
  return std::tuple<T, T>(T::random(d), T::random(d));
}

template <typename T>
void tensor_benchmarks(BenchmarkSet &set, const std::string &name) {
  using elt_t = typename T::elt_t;
  std::vector<size_t> small_sizes = make_sizes(4, 2048, 2);
  // Warm up to largest occupied memory
  warmup<T>(4194304 * 100);
  {
    BenchmarkGroup group(name + " access");
    group.add("extract_i0", extract_first_column<T>, make_two_columns<T>);
    group.add("extract_0i", extract_first_row<T>, make_two_rows<T>);
    group.add("copy_i0", copy_first_column<T>, make_two_columns<T>);
    group.add("copy_0i", copy_first_row<T>, make_two_rows<T>);
    group.add("copy_N0", copy_first_column_index<T>,
              make_two_columns_and_index<T>);
    group.add("copy_0N", copy_first_row_index<T>, make_two_rows_and_index<T>);
    group.add("const_indexed_read", vector_const_indexed_read<T>,
              make_vector_and_number<T>);
    group.add("indexed_read", vector_indexed_read<T>,
              make_vector_and_number<T>);
    group.add("indexed_write", vector_indexed_write<T>,
              make_vector_and_number<T>);
    group.add("fold_ij_jk", fold_ij_jk<T>, make_two_matrices<T>, small_sizes);
    group.add("fold_ij_kj", fold_ij_kj<T>, make_two_matrices<T>, small_sizes);
    group.add("fold_ji_jk", fold_ji_jk<T>, make_two_matrices<T>, small_sizes);
    group.add("fold_ji_kj", fold_ji_kj<T>, make_two_matrices<T>, small_sizes);
    group.add("mmult_N_N", mmult_N_N<T>, make_two_matrices<T>, small_sizes);
    group.add("mmult_T_N", mmult_T_N<T>, make_two_matrices<T>, small_sizes);
    group.add("mmult_N_T", mmult_N_T<T>, make_two_matrices<T>, small_sizes);
    group.add("mmult_T_T", mmult_T_T<T>, make_two_matrices<T>, small_sizes);
    set << group;
  }
  {
    BenchmarkGroup group(name);
    group.add("plus", add<T, T>, make_two_vectors<T>);
    group.add("minus", subtract<T, T>, make_two_vectors<T>);
    group.add("multiplies", multiply<T, T>, make_two_vectors<T>);
    group.add("divides", divide<T, T>, make_two_vectors<T>);
    group.add("plus_N", add<T, elt_t>, make_vector_and_number<T>);
    group.add("minus_N", subtract<T, elt_t>, make_vector_and_number<T>);
    group.add("multiplies_N", multiply<T, elt_t>, make_vector_and_number<T>);
    group.add("divides_N", divide<T, elt_t>, make_vector_and_number<T>);
    group.add("plus_N_inplace", add_inplace<T, elt_t>, make_vector_and_one<T>);
    group.add("minus_N_inplace", subtract_inplace<T, elt_t>,
              make_vector_and_one<T>);
    group.add("multiplies_N_inplace", multiply_inplace<T, elt_t>,
              make_vector_and_one<T>);
    group.add("divides_N_inplace", divide_inplace<T, elt_t>,
              make_vector_and_number<T>);
    group.add("sum", apply_sum<T>, make_vector<T>);
    group.add("cos", apply_cos<T>, make_vector<T>);
    group.add("exp", apply_exp<T>, make_vector<T>);
    set << group;
  }
}  // namespace benchmark

void run_all(std::ostream &out, const std::string &version = "") {
  //
  // VECTOR - VECTOR OPERATIONS
  //

  std::string name = tensor_acronym();
  if (version.size()) {
    name = name + " " + version;
  }

  auto set = BenchmarkSet(name);

  tensor_benchmarks<RTensor>(set, "RTensor");
  tensor_benchmarks<CTensor>(set, "CTensor");

  out << set << std::endl;
}

}  // namespace benchmark

/**
 * Generate a UTC ISO8601-formatted timestamp
 * and return as std::string
 */
std::string currentISO8601TimeUTC() {
  auto now = std::chrono::system_clock::now();
  auto itt = std::chrono::system_clock::to_time_t(now);
  std::ostringstream ss;
  ss << std::put_time(gmtime(&itt), "%FT%H.%MZ");
  return ss.str();
}

int main(int argn, char **argv) {
  std::string filename;
  std::string time = currentISO8601TimeUTC();
  std::string tag = "Tensor library";
  if (argn > 2) {
    tag = std::string(argv[2]);
  } else {
    tag = time;
  }
  if (argn > 1) {
    filename = argv[1];
  } else {
    filename = "benchmark_tensor_" + time + ".json";
  }
  std::cerr << "Writing output to file " << filename << std::endl;
  std::ofstream mycout(filename);
  benchmark::run_all(mycout, tag);
}
