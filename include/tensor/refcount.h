// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_REFCOUNT_H
#define TENSOR_REFCOUNT_H

#include <cstring>
#include <algorithm>

namespace tensor {

extern size_t ref_pointer_default_ref;

/**A reference counting pointer. This is a pointer that keeps track of whether
   the same pointer is shared by other structures, and when the total number of
   references drops to zero, it destroys the pointed object.

   We use this pointer to implement vectors of numbers. Two vectors can share
   the same data. Whenever one of the vectors pointing to this data is
   destroyed, the number of references drops by one. When all vectors pointing
   to the same data are destroyed, the data is, as well.

   \ingroup Internals
*/

template<class value_type>
class RefPointer {
public:
  typedef value_type elt_t; ///< Type of data pointed to

  /** Empty reference */
  RefPointer() {
    set_null();
  }

  /** Allocate a pointer of s bytes. */
  RefPointer(size_t new_size) {
    set_pointer(new elt_t[new_size], new_size);
  }

  /** Copy constructor that increases the reference count. */
  RefPointer(const RefPointer<elt_t> &p) {
    // Avoid self assignment
    if (this == &p) return;
    ref_count_ = p.ref_count_;
    data_ = p.data_;
    size_ = p.size_;
    ref();
  }

  ~RefPointer() {
    deref();
  }

  /** Ensure that we have a unique copy of the data. If the pointer has more
      than one reference, a fresh new copy of the data is created. This
      routine is useful to delay copying some data until it is really
      modified. For instance, when implementing a '+' operator between
      vectors.
  */
  void appropiate() {
    if (other_references()) {
      size_t new_size = size();
      elt_t *new_data = new elt_t[new_size];
      const elt_t *orig = constant_pointer();
      std::copy(orig, orig + new_size, new_data);
      deref();
      set_pointer(new_data, new_size);
    }
  }

  /** Replace the pointer with newly allocated data. */
  void reallocate(size_t new_size) {
    deref();
    set_pointer(new elt_t[new_size], new_size);
  }

  /** Copy a pointer increasing the reference count. */
  RefPointer<elt_t> &operator=(const RefPointer<elt_t> &p) {
    // Avoid self assignment
    if (this != &p) {
      deref();
      ref_count_ = p.ref_count_;
      data_ = p.data_;
      size_ = p.size_;
      ref();
    }
    return (*this);
  }

  /** Read/write access to the pointed data. */
  elt_t &operator[](size_t ndx) { return data_[ndx]; }

  /** Read-only access to the pointed data. */
  elt_t operator[](size_t ndx) const { return data_[ndx]; }

  /** Retreive the pointer without caring for references (unsafe). */
  elt_t *pointer() { appropiate(); return data_; }

  /** Retreive the pointer without caring for references (unsafe). */
  const elt_t *pointer() const { return data_; }

  /** Retreive the pointer without caring for references (unsafe). */
  const elt_t *constant_pointer() const { return data_; }

  /** Size of pointed-to data. */
  size_t size() const { return size_; }

  /** Reference counter */
  size_t ref_count() const { return (data_? *ref_count_ : 0); }

  /** Reference counter */
  bool other_references() const { return (data_ && (*ref_count_ > 1)); }

  /** Set to NULL */
  void reset() { deref(); set_null(); }

  /** Set to some pointer */
  void reset(elt_t *p, size_t new_size = 1) {
    deref(); set_pointer(p, new_size);
  }

private:
  // Increase the reference count of a nontrivial object
  void ref() {
    (*ref_count_)++;
  }

  // Decrease the reference count of a nontrivial object and delete the
  // memory if counter drops to zero.
  void deref() {
    if (--(*ref_count_) == 0) {
      if (data_) {
        delete[] data_;
        delete ref_count_;
      }
    }
  }

  // Unconditionally empties the structure
  void set_null() {
    size_ = 0;
    data_ = NULL;
    ref_count_ = &ref_pointer_default_ref;
  }

  // Assign a pointer creating a shared count object
  void set_pointer(elt_t *new_data, size_t new_size) {
    ref_count_ = new size_t;
    *ref_count_ = 1;
    data_ = new_data;
    size_ = new_size;
  }

  value_type *data_; // Pointer to data we reference or NULL
  size_t *ref_count_; // Number of references to that data
  size_t size_;       // Size
};

}; // namespace

#endif /* !REFCOUNT_REFCOUNT_H */
