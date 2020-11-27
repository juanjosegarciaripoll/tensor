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

#if !defined(TENSOR_TENSOR_H) || defined(TENSOR_DETAIL_TENSOR_SLICE_HPP)
#error "This header cannot be included manually"
#else
#define TENSOR_DETAIL_TENSOR_SLICE_HPP

namespace tensor {

template <typename elt_t>
class Tensor<elt_t>::view {
 public:
  ~view() { delete ranges_; };
  operator Tensor<elt_t>() const;

 private:
  const Vector<elt_t> data_;
  Indices dims_;
  Range *ranges_;

  // Start from another tensor and a set of ranges
  view(const Tensor<elt_t> &parent, Indices &dims, Range *ranges)
      : data_(parent.data_), dims_(dims), ranges_(ranges) {}

  // We do not want these objects to be initialized by users nor copied.
  view();
  view(const view &a_tensor);
  view(const Tensor<elt_t> *a_tensor, ...);
  view(Tensor<elt_t> *a_tensor, ...);

  friend class Tensor<elt_t>;
  friend class Tensor<elt_t>::mutable_view;
};

template <typename elt_t>
class Tensor<elt_t>::mutable_view {
 public:
  ~mutable_view() { delete ranges_; };

  void operator=(const view &a_stripe);
  void operator=(const Tensor<elt_t> &a_tensor);
  void operator=(elt_t v);

 private:
  Vector<elt_t> &data_;
  Indices dims_;
  Range *ranges_;

  // Start from another tensor and a set of ranges
  mutable_view(Tensor<elt_t> &parent, Indices &dims, Range *ranges)
      : data_(parent.data_), dims_(dims), ranges_(ranges) {}

  // We do not want these objects to be initialized by users nor copied.
  mutable_view();
  mutable_view(const mutable_view &a_tensor);
  mutable_view(const Tensor<elt_t> *a_tensor, ...);
  mutable_view(Tensor<elt_t> *a_tensor, ...);

  friend class Tensor<elt_t>;
};

Range *product(Range *r1, Range *r2);

}  // namespace tensor

#endif  // !TENSOR_DETAIL_TENSOR_SLICE_HPP
