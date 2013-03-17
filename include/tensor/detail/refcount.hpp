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

template<typename elt_t>
class RefPointer<elt_t>::pointer {
public:
  /* Reference counter for null pointer */
  pointer():
   data_(0), size_(0), references_(1)
  {}

  /* Reference count a given data */
  pointer(elt_t *data, size_t size) :
     data_(data), size_(size), references_(1)
  {}

  /* Create a new reference object with the same data and only 1 ro reference. */
  pointer *clone() {
    elt_t *output = new elt_t[size()];
    std::copy(begin(), end(), output);
    return new pointer(output, size());
  }

  pointer *reference() {
    ++references_;
    return this;
  }

  void dereference() {
    references_--;
    check_delete();
  }

  int references() const { return references_; }
  size_t size() { return size_; }
  elt_t *begin() { return data_; }
  elt_t *end() { return begin() + size(); }

private:

  pointer(const pointer &p); // Prevents copy constructor

  void check_delete()  {
    if (!(references_)) {
      delete[] data_;
      delete this;
    }
  }

  elt_t *data_;
  size_t size_;
  int references_;
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

template<class elt_t>
RefPointer<elt_t>::RefPointer(elt_t *data, size_t new_size) {
  ref_ = new pointer(data, new_size);
}

template<class elt_t>
RefPointer<elt_t>::RefPointer(const RefPointer<elt_t> &p) {
  ref_ = p.ref_->reference();
}

template<class elt_t>
RefPointer<elt_t>::~RefPointer() {
  ref_->dereference();
}

template<class elt_t>
void RefPointer<elt_t>::appropriate() {
  if (ref_count() > 1) {
    pointer *new_ref = ref_->clone();
    ref_->dereference();
    ref_ = new_ref;
  }
}

template<class elt_t>
void RefPointer<elt_t>::reallocate(size_t new_size) {
  ref_->dereference();
  ref_ = new pointer(new elt_t[new_size], new_size);
}

template<class elt_t>
RefPointer<elt_t> &RefPointer<elt_t>::operator=(const RefPointer<elt_t> &other) {
  ref_->dereference();
  ref_ = other.ref_->reference();
  return *this;
}

} // namespace tensor

#endif // !TENSOR_DETAIL_REFCOUNT
