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

#if !defined(TENSOR_REFCOUNT_H) || defined(TENSOR_DETAIL_REFCOUNT_HPP)
#error "This header cannot be included manually"
#else
#define TENSOR_DETAIL_REFCOUNT_HPP

namespace tensor {

//////////////////////////////////////////////////////////////////////
// REFERENCE COUNTER
//

template<typename elt_t> elt_t *clone(const elt_t *orig, size_t size) {
  elt_t *output = new elt_t[size];
  std::copy(orig, orig + size, output);
  return output;
}

template<typename elt_t> SharedPtr<elt_t>::SharedPtr(elt_t *data, size_t size) :
    data_(data), size_(size), ro_references_(1), rw_references_(0)
{}

template<typename elt_t> SharedPtr<elt_t>::SharedPtr() :
    data_(0), size_(0), ro_references_(1), rw_references_(0)
{}

template<typename elt_t> SharedPtr<elt_t> *SharedPtr<elt_t>::ro_reference() {
  if (can_mutate()) {
    return clone();
  } else {
    ++ro_references_;
    return this;
  }
}

template<typename elt_t> SharedPtr<elt_t> *SharedPtr<elt_t>::rw_reference() {
  SharedPtr *output;
  if (read_only()) {
    --ro_references_;
    output = clone();
  } else {
    output = this;
  }
  ++(output->rw_references_);
  return output;
}

template<typename elt_t> void SharedPtr<elt_t>::ro_dereference() {
  ro_references_--;
  check_delete();
}

template<typename elt_t> void SharedPtr<elt_t>::rw_dereference() {
  rw_references_--;
  check_delete();
}

template<typename elt_t>
SharedPtr<elt_t> *SharedPtr<elt_t>::clone() {
  return new SharedPtr(tensor::clone(data_, size_), size_);
}

template<typename elt_t>
void SharedPtr<elt_t>::check_delete() {
  if (!(ro_references_ | rw_references_)) {
    delete[] data_;
    delete this;
  }
}

//////////////////////////////////////////////////////////////////////
// SHARED POINTER WITH COPY ON WRITE
//

template<class elt_t>
RefPointer<elt_t>::RefPointer() {
  ref_ = new SharedPtr<elt_t>();
}

template<class elt_t>
RefPointer<elt_t>::RefPointer(size_t new_size) {
  ref_ = new SharedPtr<elt_t>(new elt_t[new_size], new_size);
}

template<class elt_t>
RefPointer<elt_t>::RefPointer(elt_t *data, size_t new_size) {
  ref_ = new SharedPtr<elt_t>(data, new_size);
}

template<class elt_t>
RefPointer<elt_t>::RefPointer(const RefPointer<elt_t> &p) {
  ref_ = p.ref_->ro_reference();
}

template<class elt_t>
RefPointer<elt_t>::~RefPointer() {
  ref_->ro_dereference();
}

template<class elt_t>
void RefPointer<elt_t>::appropiate() {
  if (ref_->read_only()) {
    SharedPtr<elt_t> *new_ref = ref_->clone();
    ref_->ro_dereference();
    ref_ = new_ref;
  }
}

template<class elt_t>
void RefPointer<elt_t>::reallocate(size_t new_size) {
  reset(new elt_t[new_size], new_size);
}

template<class elt_t>
RefPointer<elt_t> &RefPointer<elt_t>::operator=(const RefPointer<elt_t> &other) {
  ref_->ro_dereference();
  ref_ = other.ref_->ro_reference();
  return *this;
}

template<class elt_t>
void RefPointer<elt_t>::reset(elt_t *p, size_t new_size) {
  ref_->ro_dereference();
  ref_ = new SharedPtr<elt_t>(p, new_size);
}

template<typename elt_t>
SharedPtr<elt_t> *RefPointer<elt_t>::rw_reference() const {
  return (ref_ = ref_->rw_reference());
}

} // namespace tensor

#endif // !TENSOR_DETAIL_REFCOUNT
