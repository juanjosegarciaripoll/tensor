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

#ifndef TENSOR_PROFILE_PROFILE_H
#define TENSOR_PROFILE_PROFILE_H

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <ctime>
#include <functional>
#include <tensor/tools.h>
#include <tensor/config.h>

namespace benchmark {

static size_t __count_executions = 0;

template <typename T>
void force(const T &t) {
  __count_executions += t.size() != 0;
}

template <typename T>
void force_nonzero(const T &t) {
  __count_executions += (t != static_cast<T>(0));
}

struct BenchmarkSet;
struct BenchmarkGroup;
struct BenchmarkItem;

std::string tensor_acronym() {
  std::string compiler =
#if defined(_MSC_VER)
      "MSVC-";
#elif defined(__GNUC__)
      "GCC-";
#elif defined(__clang__)
      "Clang-";
#else
      "";
#endif
  std::string platform =
#if defined(_WIN64)
      "W64-";
#elif defined(_WIN32)
      "W32-";
#elif defined(__linux__)
      "Linux-";
#elif defined(__APPLE__)
      "Darwin-";
#else
          "";
#endif
  std::string blas_library =
#if defined(TENSOR_USE_ATLAS)
      "Atlas";
#elif defined(TENSOR_USE_OPENBLAS)
      "OpenBLAS";
#elif defined(TENSOR_USE_VECLIB)
      "Veclib";
#elif defined(TENSOR_USE_MKL)
      "MKL";
#elif defined(TENSOR_USE_ACML)
      "ACML";
#elif defined(TENSOR_USE_ESSL)
              "ESSL";
#elif defined(TENSOR_USE_CBLAPACK)
              "CBLAPACK";
#else
              "BLAS";
#endif
  return std::string("tensor ") + compiler + platform + blas_library;
}

std::string tensor_environment() {
  std::string compiler =
#if defined(_MSC_VER)
      "Microsoft C++";
#elif defined(__GNUC__)
      "Gnu C++";
#elif defined(__clang__)
      "Clang C++";
#else
      "Unknown C++ compiler";
#endif
  std::string platform =
#if defined(_WIN64)
      "Windows AMD64";
#elif defined(__linux__)
      "Linux";
#elif defined(__APPLE__)
      "Darwin";
#else
      "Unknown OS";
#endif
  std::string blas_library =
#if defined(TENSOR_USE_ATLAS)
      "Atlas";
#elif defined(TENSOR_USE_OPENBLAS)
      "OpenBLAS";
#elif defined(TENSOR_USE_VECLIB)
      "Apple Veclib";
#elif defined(TENSOR_USE_MKL)
      "Intel MKL";
#elif defined(TENSOR_USE_ACML)
              "ACML";
#elif defined(TENSOR_USE_ESSL)
              "IBM ESSL";
#elif defined(TENSOR_USE_CBLAPACK)
              "CBLAPACK";
#else
#error "Unknown BLAS library"
#endif
  return compiler + ", " + platform + ", " + blas_library;
}

struct BenchmarkSet {
  std::string name = tensor_acronym();
  std::string environment = tensor_environment();
  std::vector<BenchmarkGroup> groups{};

  BenchmarkSet(const std::string &aname) : name(aname), groups{} {
    std::cerr << "===================\nStarting set " << name << '\n'
              << "Environment: " << environment << '\n';
  }

  BenchmarkSet &operator<<(const BenchmarkGroup &group) {
    groups.push_back(group);
    return *this;
  }
};

struct BenchmarkGroup {
  std::string name{};
  std::vector<BenchmarkItem> items{};

  BenchmarkGroup(const std::string &aname) : name(aname), items{} {
    std::cerr << "------------------\nStarting group " << name << '\n';
  }

  BenchmarkGroup &operator<<(const BenchmarkItem &item) {
    items.push_back(item);
    return *this;
  }

  template <typename run, typename setup>
  BenchmarkGroup &add(const char *aname, run f, setup s,
                      const std::vector<size_t> &sizes = {}) {
    items.push_back(BenchmarkItem(aname, f, s, sizes));
    return *this;
  }
};

std::vector<size_t> make_sizes(size_t start, size_t end, size_t factor = 2) {
  std::vector<size_t> output;
  while (start <= end) {
    output.push_back(start);
    start *= factor;
  }
  return output;
}

struct BenchmarkItem {
  std::string name{};
  std::vector<size_t> sizes{};
  std::vector<double> times{};

  static std::vector<size_t> default_sizes() {
    return make_sizes(1, 4194304, 4);
  };

  template <typename Functor>
  static double timeit(Functor f, size_t repeats) {
    tensor::tic();
    for (size_t j = 0; j < repeats; ++j) {
      f();
    }
    return tensor::toc();
  }

  template <typename Functor>
  static double autorange(Functor f, double limit = 0.2) {
    size_t repeats = 1;
    double time = 0.0;
    std::cerr.precision(17);
    for (int attempts = 4; attempts; --attempts) {
      time = timeit(f, repeats);
      if (time >= limit) {
        break;
      }
      repeats =
		static_cast<size_t>(1.5 * limit * static_cast<double>(repeats) / std::max(time, 1e-8));
    }
    return time / static_cast<double>(repeats);
  }

  template <class args_tuple>
  BenchmarkItem(const std::string &aname, void (*f)(args_tuple &),
                args_tuple (*s)(size_t), const std::vector<size_t> &asizes = {})
      : name(aname),
        sizes(asizes.size() ? asizes : default_sizes()),
        times(sizes.size()) {
    for (size_t i = 0; i < sizes.size(); i++) {
      args_tuple args = s(sizes[i]);
      times[i] = autorange([&]() { f(args); });
      std::cerr << "Executing item " << name << " at size " << sizes[i]
                << " took " << times[i] << " seconds per iteration\n";
    }
  }
};

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
  const char *comma = "";
  out << '[';
  for (auto &item : v) {
    out << comma << item;
    comma = ",";
  }
  out << ']';
  return out;
}

std::ostream &operator<<(std::ostream &out, const BenchmarkSet &set) {
  out << "{\"name\": \"" << set.name << "\", \"environment\": \""
      << set.environment << "\", \"groups\": " << set.groups << "}";
  return out;
}

std::ostream &operator<<(std::ostream &out, const BenchmarkGroup &group) {
  out << "{\"name\": \"" << group.name << "\", \"items\": " << group.items
      << "}";
  return out;
}

std::ostream &operator<<(std::ostream &out, const BenchmarkItem &item) {
  out << "{\"name\": \"" << item.name << "\", \"sizes\": " << item.sizes
      << ", \"times\": " << item.times << "}";
  return out;
}

}  // namespace benchmark

#endif  // TENSOR_PROFILE_PROFILE_H
