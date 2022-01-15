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
#include <functional>
#include <iostream>
#include <fstream>
#include <tuple>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <memory>

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

template <class T>
void warmup(size_t size) {
  for (int i = 0; i < 10; ++i) {
    std::unique_ptr<T> p{new T(size)};
  }
}

template <class T>
std::tuple<T, typename T::elt_t> make_vector_and_number(size_t size) {
  typedef typename T::elt_t Number;
  auto number = static_cast<Number>(3.0);
  warmup<T>(size);
  return typename std::tuple<T, Number>(T::random(size), number);
}
template <class T>
std::tuple<T, T> make_two_vectors(size_t size) {
  warmup<T>(size);
  return std::tuple<T, T>(T::random(size), T::random(size) + 1.0);
}

void run_all(std::ostream &out, const std::string &version = "") {
  //
  // VECTOR - VECTOR OPERATIONS
  //
  std::vector<size_t> sizes{1,    4,     16,    64,     256,     1024,
                            4096, 16384, 65536, 262144, 1048576, 4194304};

  std::string name = "tensor library";
  if (version.size()) {
    name = name + " " + version;
  }

  auto set = BenchmarkSet(name);

  set << BenchmarkGroup("RTensor")
             .add("plus", add<RTensor, RTensor>, make_two_vectors<RTensor>,
                  sizes)
             .add("minus", subtract<RTensor, RTensor>,
                  make_two_vectors<RTensor>, sizes)
             .add("multiplies", multiply<RTensor, RTensor>,
                  make_two_vectors<RTensor>, sizes)
             .add("divides", divide<RTensor, RTensor>,
                  make_two_vectors<RTensor>, sizes);
  set << BenchmarkGroup("CTensor")
             .add("plus", add<CTensor, CTensor>, make_two_vectors<CTensor>,
                  sizes)
             .add("minus", subtract<CTensor, CTensor>,
                  make_two_vectors<CTensor>, sizes)
             .add("multiplies", multiply<CTensor, CTensor>,
                  make_two_vectors<CTensor>, sizes)
             .add("divides", divide<CTensor, CTensor>,
                  make_two_vectors<CTensor>, sizes);
  set << BenchmarkGroup("RTensor with number")
             .add("plusN", add<RTensor, double>,
                  make_vector_and_number<RTensor>, sizes)
             .add("minusN", subtract<RTensor, double>,
                  make_vector_and_number<RTensor>, sizes)
             .add("multipliesN", multiply<RTensor, double>,
                  make_vector_and_number<RTensor>, sizes)
             .add("dividesN", divide<RTensor, double>,
                  make_vector_and_number<RTensor>, sizes);
  set << BenchmarkGroup("CTensor with number")
             .add("plusN", add<CTensor, cdouble>,
                  make_vector_and_number<CTensor>, sizes)
             .add("minusN", subtract<CTensor, cdouble>,
                  make_vector_and_number<CTensor>, sizes)
             .add("multipliesN", multiply<CTensor, cdouble>,
                  make_vector_and_number<CTensor>, sizes)
             .add("dividesN", divide<CTensor, cdouble>,
                  make_vector_and_number<CTensor>, sizes);

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
