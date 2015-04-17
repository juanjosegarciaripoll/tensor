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

#include <tensor/numbers.h>

namespace tensor {

template<typename elt_t>
class RefPointer<elt_t>::pointer {
public:
  /* Reference counter for null pointer */
  pointer():
    data_(0), size_(0), references_(1), owned_(true)
  {}

  /* Reference count a given data */
  pointer(elt_t *data, size_t size, bool owned = true) :
    data_(data), size_(size), references_(1), owned_(owned)
  {}

  /* Delete the object and its data */
  ~pointer() {
    if (owned_)
      delete[] data_;
  }

  /* Create a new reference object with the same data and only 1 ro reference. */
  pointer *clone() {
    elt_t *output = new elt_t[size()];
    std::copy(begin(), end(), output);
    return new pointer(output, size());
  }

  int reference() { return ++references_; }
  int dereference() { return --references_; }
  int references() const { return references_; }
  size_t size() { return size_; }
  elt_t *begin() { return data_; }
  elt_t *end() { return begin() + size(); }

private:

  pointer(const pointer &p); // Prevents copy constructor

  elt_t *data_;
  size_t size_;
  int references_;
  bool owned_;
};

//////////////////////////////////////////////////////////////////////
// SHARED POINTER WITH COPY ON WRITE
//

template<class elt_t>
RefPointer<elt_t>::RefPointer() {
  ref_ = new pointer();
}

template<class elt_t>
RefPointer<elt_t>::RefPointer(size_t new_size) {
  ref_ = new pointer(new elt_t[new_size], new_size);
}

/* This is an optimization to ensure that complex double vectors are
 * started uninitialized. */
template<>
inline RefPointer<cdouble>::RefPointer(size_t new_size) {
  ref_ = new pointer(reinterpret_cast<cdouble*>(new double[2*new_size]), new_size);
}

template<>
inline RefPointer<cdouble>::pointer *
RefPointer<cdouble>::pointer::clone() {
  cdouble *output = reinterpret_cast<cdouble*>(new double[2*size()]);
  memcpy(output, begin(), sizeof(*output)*size());
  return new pointer(output, size());
}

template<class elt_t>
RefPointer<elt_t>::RefPointer(elt_t *data, size_t new_size, bool owned) {
    ref_ = new pointer(data, new_size, owned);
}

template<class elt_t>
RefPointer<elt_t>::RefPointer(const RefPointer<elt_t> &p) {
  ref_ = p.reference();
}

template<class elt_t>
RefPointer<elt_t>::~RefPointer() {
  dereference();
}

template<class elt_t>
typename RefPointer<elt_t>::pointer *RefPointer<elt_t>::reference() const {
  ref_->reference();
  return ref_;
}

template<class elt_t>
void RefPointer<elt_t>::dereference() {
  if (ref_->dereference() <= 0)
    delete ref_;
}

template<class elt_t>
void RefPointer<elt_t>::appropriate() {
  if (ref_count() > 1) {
    pointer *new_ref = ref_->clone();
    dereference();
    ref_ = new_ref;
  }
}

template<class elt_t>
void RefPointer<elt_t>::reallocate(size_t new_size) {
  dereference();
  ref_ = new pointer(new elt_t[new_size], new_size);
}

template<class elt_t>
RefPointer<elt_t> &RefPointer<elt_t>::operator=(const RefPointer<elt_t> &other) {
  if (other.ref_ != ref_) {
    dereference();
    ref_ = other.reference();
  }
  return *this;
}

} // namespace tensor

#endif // !TENSOR_DETAIL_REFCOUNT
