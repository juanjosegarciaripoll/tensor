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
#ifndef TENSOR_DETAIL_INITIALIZER_H
#include <utility>
#include <tensor/indices.h>

/** \cond */

namespace tensor {

template <typename elt_t>
class Tensor;

namespace detail {

template <size_t rank, typename item>
struct nested_initializer_list {
  typedef std::initializer_list<
      typename nested_initializer_list<rank - 1, item>::type>
      type;
};

template <typename item>
struct nested_initializer_list<1, item> {
  typedef std::initializer_list<item> type;
};

template <typename elt_t>
class nested_list_initializer {
  template <typename item>
  struct nl_rank {
    static constexpr size_t value = 0;
  };

  template <typename item>
  struct nl_rank<std::initializer_list<item>> {
    static constexpr size_t value = nl_rank<item>::value + 1;
  };

  template <class item>
  static constexpr size_t get_dimension(std::initializer_list<item> l,
                                        size_t ndx) {
    return ndx ? (l.size() ? get_dimension(*l.begin(), ndx - 1) : 0) : l.size();
  }

  template <class item>
  static constexpr size_t get_dimension(item, size_t) {
    return 0;
  }

  template <typename nl, size_t... I>
  static constexpr auto make_dimensions_inner(nl l, std::index_sequence<I...>) {
    return Dimensions{get_dimension(l, I)...};
  }

  static inline void copy_into(elt_t *buffer, const index *, index,
                               const elt_t &value) {
    *buffer = value;
  }

  template <typename item>
  static void copy_into(elt_t *buffer, const index *dims, index stride,
                        const std::initializer_list<item> &l) {
    if (*dims != static_cast<index>(l.size())) {
      throw std::out_of_range(
          "Mismatch between tensor initializer list and dimensions.");
    }
    const index next_stride = stride * (*dims);
    for (auto x : l) {
      copy_into(buffer, dims + 1, next_stride, x);
      buffer += stride;
    }
  }

 public:
  template <typename nl>
  static constexpr auto dimensions(nl l) {
    return make_dimensions_inner(
        l, std::make_index_sequence<nl_rank<decltype(l)>::value>());
  }

  template <typename nl>
  static constexpr void copy_into(elt_t *buffer, const Dimensions &dims,
                                  const nl &l) {
    return copy_into(buffer, dims.begin(), 1, l);
  }

  static constexpr Tensor<elt_t> make_tensor(
      typename nested_initializer_list<1, elt_t>::type l) {
    auto output = Tensor<elt_t>(dimensions(l));
    copy_into(output.begin(), output.dimensions().begin(), 1, l);
    return output;
  }
  static constexpr Tensor<elt_t> make_tensor(
      typename nested_initializer_list<2, elt_t>::type l) {
    auto output = Tensor<elt_t>(dimensions(l));
    copy_into(output.begin(), output.dimensions().begin(), 1, l);
    return output;
  }
  static constexpr Tensor<elt_t> make_tensor(
      typename nested_initializer_list<3, elt_t>::type l) {
    auto output = Tensor<elt_t>(dimensions(l));
    copy_into(output.begin(), output.dimensions().begin(), 1, l);
    return output;
  }
  static constexpr Tensor<elt_t> make_tensor(
      typename nested_initializer_list<4, elt_t>::type l) {
    auto output = Tensor<elt_t>(dimensions(l));
    copy_into(output.begin(), output.dimensions().begin(), 1, l);
    return output;
  }
  static constexpr Tensor<elt_t> make_tensor(
      typename nested_initializer_list<5, elt_t>::type l) {
    auto output = Tensor<elt_t>(dimensions(l));
    copy_into(output.begin(), output.dimensions().begin(), 1, l);
    return output;
  }
  static constexpr Tensor<elt_t> make_tensor(
      typename nested_initializer_list<6, elt_t>::type l) {
    auto output = Tensor<elt_t>(dimensions(l));
    copy_into(output.begin(), output.dimensions().begin(), 1, l);
    return output;
  }
};

/** \endcond */

}  // namespace detail

}  // namespace tensor

#endif  // TENSOR_DETAIL_INITIALIZER_H