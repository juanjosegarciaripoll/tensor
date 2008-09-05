// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_REFCOUNT_H
#define TENSOR_REFCOUNT_H

#include <cstring>

#ifdef REFCOUNT_USE_STATIC_COUNTER
# define REFCOUNT_DEFAULT_COUNTER &ref_pointer_default_ref;
# define REFCOUNT_HAS_COUNTER 1
extern size_t ref_pointer_default_ref;
#else
# define REFCOUNT_DEFAULT_COUNTER NULL
# define REFCOUNT_HAS_COUNTER ref_count_
#endif

/**A reference counting pointer. This is a pointer that keeps track of whether
   the same pointer is shared by other structures, and when the total number of
   references drops to zero, it destroys the pointed object.

   We use this pointer to implement vectors of numbers. Two vectors can share
   the same data. Whenever one of the vectors pointing to this data is
   destroyed, the number of references drops by one. When all vectors pointing
   to the same data are destroyed, the data is, as well.

   \ingroup Internals
*/

template<class value_type, bool simple_copy = true>
class RefPointer {
public:
  typedef value_type elt_t; ///< Type of data pointed to

  /** Allocate a pointer of s bytes. */
  RefPointer(size_t s = 0) {
    new_data(s);
  }

  /** Copy constructor that increases the reference count. */
  RefPointer(const RefPointer<elt_t,simple_copy> &p) {
    // Avoid self assignment
    if (this == &p) return;
    ref_count_ = p.ref_count_;
    __data = p.__data;
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
    if (REFCOUNT_HAS_COUNTER && (*ref_count_ > 1)) {
      RefPointer<elt_t,simple_copy> aux(size_);
      if (simple_copy) {
	std::memcpy(aux.__data, __data, sizeof(elt_t) * size_);
      } else {
	for (size_t i = 0; i < size_; i++) {
	  aux.__data[i] = __data[i];
	}
      }
      *this = aux;
    }
  }

  /** Replace the pointer with newly allocated data. */
  void reallocate(size_t new_size) {
    deref();
    new_data(new_size);
  }

  /** Copy a pointer increasing the reference count. */
  RefPointer<elt_t,simple_copy>
  &operator=(const RefPointer<elt_t,simple_copy> &p) {
    // Avoid self assignment
    if (this != &p) {
      deref();
      ref_count_ = p.ref_count_;
      __data = p.__data;
      size_ = p.size_;
      ref();
    }
    return (*this);
  }

  /** Read/write access to the pointed data. */
  elt_t &operator[](size_t ndx) { return __data[ndx]; }

  /** Read-only access to the pointed data. */
  elt_t operator[](size_t ndx) const { return __data[ndx]; }

  /** Retreive the pointer without caring for references (unsafe). */
  elt_t *pointer() { appropiate(); return __data; }

  /** Retreive the pointer without caring for references (unsafe). */
  const elt_t *pointer() const { return __data; }

  /** Retreive the pointer without caring for references (unsafe). */
  const elt_t *constant_pointer() const { return __data; }

  /** Size of pointed-to data. */
  size_t size() const { return size_; }

  /** Reference counter */
  size_t ref_count() const { return (REFCOUNT_HAS_COUNTER? *ref_count_ : 0); }

  /** Coercion to pointer */
  elt_t *operator *() { return pointer(); }

#ifndef REFCOUNT_USE_STATIC_COUNTER
  /** Point to preallocated data */
  void set_pointer(size_t new_size, elt_t *p) {
    deref(); __data = p; size_ = new_size; ref_count_ = 0;
  }
#endif

private:
  // Increase the reference count of a nontrivial object
  void ref() {
    if (REFCOUNT_HAS_COUNTER)
      (*ref_count_)++;
  }

  // Decrease the reference count of a nontrivial object and delete the
  // memory if counter drops to zero.
  void deref() {
    if (REFCOUNT_HAS_COUNTER) {
      if (--(*ref_count_) == 0) {
	delete[] __data;
	delete ref_count_;
      }
    }
    __data = NULL;
    ref_count_ = REFCOUNT_DEFAULT_COUNTER;
    size_ = 0;
  }

  // Allocate a C array with the given size and assign it to us
  void new_data(size_t new_size) {
    size_ = new_size;
    if (new_size == 0) {
      ref_count_ = REFCOUNT_DEFAULT_COUNTER;
      __data = NULL;
      ref();
    } else {
      // We are the only owners of the data
      ref_count_ = new size_t;
      *ref_count_ = 1;
      __data = new elt_t[new_size];
    }
  }

  value_type *__data; // Pointer to data we reference or NULL
  size_t *ref_count_; // Number of references to that data
  size_t size_;       // Size
};

#endif /* !REFCOUNT_REFCOUNT_H */
