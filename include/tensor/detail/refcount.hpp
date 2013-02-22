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

/**A reference counting object. This object keeps a pointer and the number
   of references to it which are read/only or read/write. A pointer can
   be shared by multiple read/only or by multiple read/write references,
   but if there are read/write references, only one object can be tagged
   as read/only reference.
*/
template<typename elt_t>
class RefPointer<elt_t>::pointer {
public:
  /** Reference counter for null pointer */
  pointer():
   data_(0), size_(0), ro_references_(1), rw_references_(0)
  {}

  /** Reference count a given data */
  pointer(elt_t *data, size_t size) :
     data_(data), size_(size), ro_references_(1), rw_references_(0)
  {}

  /** Create a new reference object with the same data and only 1 ro reference. */
  pointer *clone() {
    elt_t *output = new elt_t[size()];
    std::copy(begin(), end(), output);
    return new pointer(output, size());
  }

  /** Add a read/only reference. */
  pointer *ro_reference() {
    if (can_mutate()) {
      return clone();
    } else {
      ++ro_references_;
      return this;
    }
  }

  /** Add a read/write reference. */
  pointer *rw_reference() {
    pointer *output;
    if (read_only()) {
      --ro_references_;
      output = clone();
    } else {
      output = this;
    }
    ++(output->rw_references_);
    return output;
  }

  /** Eliminate a read/only reference. */
  void ro_dereference() {
    ro_references_--;
    check_delete();
  }

  /** Eliminate a read/write reference. */
  void rw_dereference() {
    rw_references_--;
    check_delete();
  }

  /** An object is read/only if other references expect it not to change. */
  bool read_only() { return ro_references() > 1; }

  /** An object can mutate if there are references that can change it. */
  bool can_mutate() { return rw_references() > 0; }

  /** Return number of read/only references. */
  int ro_references() { return ro_references_; }

  /** Return number of read/write references. */
  int rw_references() { return rw_references_; }

  /** Amount of data allocated. */
  size_t size() { return size_; }

  /** Return pointer. */
  elt_t *begin() { return data_; }

  /** Return end pointer. */
  elt_t *end() { return begin() + size(); }

private:

  pointer(const pointer &p); // Prevents copy constructor

  void check_delete()  {
    if (!(ro_references_ | rw_references_)) {
      delete[] data_;
      delete this;
    }
  }

  elt_t *data_;
  size_t size_;
  int ro_references_, rw_references_;
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
  ref_ = p.ref_->ro_reference();
}

template<class elt_t>
RefPointer<elt_t>::~RefPointer() {
  ref_->ro_dereference();
}

template<class elt_t>
void RefPointer<elt_t>::appropiate() {
  if (ref_->read_only()) {
    pointer *new_ref = ref_->clone();
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
  ref_ = new pointer(p, new_size);
}

} // namespace tensor

#endif // !TENSOR_DETAIL_REFCOUNT
