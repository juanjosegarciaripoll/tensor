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
std::tuple<T, typename T::elt_t> make_vector_and_number(size_t size) {
  return typename std::tuple<T, typename T::elt_t>(T::random(size),
                                                   T::random(1)[0] + 1.0);
}
template <class T>
std::tuple<T, T> make_two_vectors(size_t size) {
  return std::tuple<T, T>(T::random(size), T::random(size) + 1.0);
}

void run_all(std::ostream &out) {
  //
  // VECTOR - VECTOR OPERATIONS
  //
  std::vector<size_t> sizes{1,    4,     16,    64,     256,     1024,
                            4096, 16384, 65536, 262144, 1048576, 4194304};

  auto set = BenchmarkSet("tensor library");

  set << (BenchmarkGroup("RTensor")
          << BenchmarkItem("plus", add<RTensor, RTensor>,
                           make_two_vectors<RTensor>, sizes)
          << BenchmarkItem("minus", subtract<RTensor, RTensor>,
                           make_two_vectors<RTensor>, sizes)
          << BenchmarkItem("multiplies", multiply<RTensor, RTensor>,
                           make_two_vectors<RTensor>, sizes)
          << BenchmarkItem("divides", multiply<RTensor, RTensor>,
                           make_two_vectors<RTensor>, sizes))
      << (BenchmarkGroup("CTensor")
          << BenchmarkItem("plus", add<CTensor, CTensor>,
                           make_two_vectors<CTensor>, sizes)
          << BenchmarkItem("minus", subtract<CTensor, CTensor>,
                           make_two_vectors<CTensor>, sizes)
          << BenchmarkItem("multiplies", multiply<CTensor, CTensor>,
                           make_two_vectors<CTensor>, sizes)
          << BenchmarkItem("divides", multiply<CTensor, CTensor>,
                           make_two_vectors<CTensor>, sizes))
      << (BenchmarkGroup("RTensor")
          << BenchmarkItem("plusN", add<RTensor, double>,
                           make_vector_and_number<RTensor>, sizes)
          << BenchmarkItem("minusN", subtract<RTensor, double>,
                           make_vector_and_number<RTensor>, sizes)
          << BenchmarkItem("multipliesN", multiply<RTensor, double>,
                           make_vector_and_number<RTensor>, sizes)
          << BenchmarkItem("dividesN", multiply<RTensor, double>,
                           make_vector_and_number<RTensor>, sizes))
      << (BenchmarkGroup("CTensor")
          << BenchmarkItem("plusN", add<CTensor, cdouble>,
                           make_vector_and_number<CTensor>, sizes)
          << BenchmarkItem("minusN", subtract<CTensor, cdouble>,
                           make_vector_and_number<CTensor>, sizes)
          << BenchmarkItem("multipliesN", multiply<CTensor, cdouble>,
                           make_vector_and_number<CTensor>, sizes)
          << BenchmarkItem("dividesN", multiply<CTensor, cdouble>,
                           make_vector_and_number<CTensor>, sizes));

  out << set << std::endl;
}

int main(int argn, char **argv) {
  if (argn > 1) {
    std::cerr << "Writing output to file " << argv[1] << std::endl;
    std::ofstream mycout(argv[1]);
    run_all(mycout);
  } else {
    run_all(std::cout);
  }
}
