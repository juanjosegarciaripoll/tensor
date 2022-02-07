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

#include <tensor/tensor.h>
#include <iostream>
#include <fstream>
#include <tuple>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <tensor/io.h>

#include "profile.h"

using namespace tensor;
using namespace benchmark;

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
  for (tensor::index i = 0; i < v.size(); ++i) {
    x += v[i];
  }
  force_nonzero(x);
}

template <class T>
void vector_indexed_read(std::tuple<T, typename T::elt_t> &args) {
  T &v = std::get<0>(args);
  auto n = std::get<1>(args);
  auto x = n;
  for (tensor::index i = 0; i < v.size(); ++i) {
    x += v[i];
  }
  force_nonzero(x);
}

template <class T>
void vector_indexed_write(std::tuple<T, typename T::elt_t> &args) {
  T &v = std::get<0>(args);
  auto n = std::get<1>(args);
  for (tensor::index i = 0; i < v.size(); ++i) {
    v.at(i) = n;
  }
}

template <class T>
void copy_first_column(std::tuple<T, T> &args) {
  T &A = std::get<0>(args);
  T &B = std::get<1>(args);
  A.at(range(), range(0)) = B(range(), range(0));
}

template <class T>
void copy_first_row(std::tuple<T, T> &args) {
  T &A = std::get<0>(args);
  T &B = std::get<1>(args);
  A.at(range(0), range()) = B(range(0), range());
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
  warmup<T>(size);
  return typename std::tuple<T, typename T::elt_t>(
      T::random(static_cast<tensor::index>(size)), number);
}

template <class T>
std::tuple<T, T> make_two_vectors(size_t size) {
  warmup<T>(size);
  return std::tuple<T, T>(T::random(static_cast<tensor::index>(size)),
                          T::random(static_cast<tensor::index>(size)) + 1.0);
}

template <class T>
std::tuple<T> make_vector(size_t size) {
  warmup<T>(size);
  return typename std::tuple<T>(T::random(static_cast<tensor::index>(size)));
}

template <class T>
std::tuple<T, typename T::elt_t> make_vector_and_one(size_t size) {
  auto number = static_cast<typename T::elt_t>(1.0);
  warmup<T>(size);
  return typename std::tuple<T, typename T::elt_t>(
      T::random(static_cast<tensor::index>(size)), number);
}

template <class T>
std::tuple<T, T> make_two_matrices(size_t size) {
  warmup<T>(size);
  size = static_cast<size_t>(sqrt(static_cast<double>(size)));
  Dimensions d = {static_cast<tensor::index>(size),
                  static_cast<tensor::index>(size)};
  return std::tuple<T, T>(T::random(d), T::random(d));
}

template <typename T>
void tensor_benchmarks(BenchmarkSet &set, const std::string &name) {
  typedef typename T::elt_t elt_t;

  {
    BenchmarkGroup group(name);
    group.add("copy_column", copy_first_column<T>, make_two_matrices<T>);
    group.add("copy_row", copy_first_row<T>, make_two_matrices<T>);
    group.add("plus", add<T, T>, make_two_vectors<T>);
    group.add("minus", subtract<T, T>, make_two_vectors<T>);
    group.add("multiplies", multiply<T, T>, make_two_vectors<T>);
    group.add("divides", divide<T, T>, make_two_vectors<T>);
    group.add("sum", apply_sum<T>, make_vector<T>);
    group.add("cos", apply_cos<T>, make_vector<T>);
    group.add("exp", apply_exp<T>, make_vector<T>);
    set << group;
  }
  {
    BenchmarkGroup group(name + " access");
    group.add("const_indexed_read", vector_const_indexed_read<T>,
              make_vector_and_number<T>);
    group.add("indexed_read", vector_indexed_read<T>,
              make_vector_and_number<T>);
    group.add("indexed_write", vector_indexed_write<T>,
              make_vector_and_number<T>);
    set << group;
  }
  {
    BenchmarkGroup group(name + " with number");
    group.add("plusN", add<T, elt_t>, make_vector_and_number<T>);
    group.add("minusN", subtract<T, elt_t>, make_vector_and_number<T>);
    group.add("multipliesN", multiply<T, elt_t>, make_vector_and_number<T>);
    group.add("dividesN", divide<T, elt_t>, make_vector_and_number<T>);
    group.add("plusNinplace", add_inplace<T, elt_t>, make_vector_and_one<T>);
    group.add("minusNinplace", subtract_inplace<T, elt_t>,
              make_vector_and_one<T>);
    group.add("multipliesNinplace", multiply_inplace<T, elt_t>,
              make_vector_and_one<T>);
    group.add("dividesNinplace", divide_inplace<T, elt_t>,
              make_vector_and_number<T>);
    set << group;
  }
}

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
  run_all(mycout, tag);
}
