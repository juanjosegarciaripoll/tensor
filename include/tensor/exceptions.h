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

#ifndef TENSOR_EXCEPTIONS_H
#define TENSOR_EXCEPTIONS_H

#include <stdexcept>
#include <tensor/numbers.h>

namespace tensor {

#if __cplusplus >= 202002L
#define tensor_unlikely(x) (x) [[unlikely]]
#define tensor_likely(x) (x) [[likely]]
#else
#if defined(__GNUC__) || defined(__clang__)
#define tensor_unlikely(x) (__builtin_expect((x), 0))
#define tensor_likely(x) (__builtin_expect((x), 1))
#else
#define tensor_unlikely(x) (x)
#define tensor_likely(x) (x)
#endif
#endif

struct invalid_assertion : public std::logic_error {
  invalid_assertion(const char *message, const char *a_filename, int a_line)
      : std::logic_error(message), filename{a_filename}, line{a_line} {}
  const char *filename = "";
  int line{};
};

struct invalid_dimension : public std::invalid_argument {
  invalid_dimension()
      : std::invalid_argument(
            "Invalid dimension size (negative or too large)"){};
};

struct out_of_bounds_index : public std::out_of_range {
  out_of_bounds_index()
      : std::out_of_range("Index out of tensor dimension's bounds"){};
};

struct iterator_overflow : public std::out_of_range {
  iterator_overflow()
      : std::out_of_range("Iterator accessed elements out of the tensor"){};
};

class Dimensions;

struct dimensions_mismatch : public std::out_of_range {
  dimensions_mismatch() : std::out_of_range("Mismatch in tensor dimensions"){};
  explicit dimensions_mismatch(const char *message)
      : std::out_of_range(message){};
  dimensions_mismatch(const Dimensions &d1, const Dimensions &d2);
  dimensions_mismatch(const Dimensions &d1, const Dimensions &d2, index which1,
                      index which2);
};

[[noreturn]] void tensor_terminate(const std::exception &exception);
[[noreturn]] void tensor_terminate(const invalid_assertion &exception);

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define tensor_expects(expression) tensor_assert(expression)
#define tensor_assert(assertion) \
  tensor_assert2((assertion),    \
                 ::tensor::invalid_assertion(#assertion, __FILE__, __LINE__))
#ifdef TENSOR_DEBUG
#define tensor_assert2(expression, condition) \
  if (!(expression)) {                        \
    ::tensor::tensor_terminate(condition);    \
  }
#define tensor_noexcept noexcept
#else
#define tensor_assert2(expression, condition)
#define tensor_noexcept noexcept
#endif
// NOLINTEND(cppcoreguidelines-macro-usage)

// narrow_cast(): a searchable way to do narrowing casts of values
template <class T, class U>
constexpr T narrow_cast(U &&u) noexcept {
  return static_cast<T>(std::forward<U>(u));
}

}  // namespace tensor

#endif  // TENSOR_EXCEPTIONS_H
