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

#pragma once
#ifndef TENSOR_REFCOUNT_H
#define TENSOR_REFCOUNT_H

#include <cstring>
#include <algorithm>

namespace tensor {

/**A reference counting pointer with copy-on-write. This is a pointer that keeps
   track of whether the same data is shared by other RefPointer structures. It
   internally keeps a reference counter to store how many pointers look at the
   data. When the total number of references drops to zero, it destroys the
   pointed object.

   This allows us to have different parts of the code use the same physical
   (tensor) data, thus avoiding expensive copy operations. The RefPointer
   behaves thus similar to pointers, but with internal bookkeeping and easier use.

   As soon as you manipulate the data (access it through a non-const pointer),
   it will be silently and transparently copied to another location if the data
   is shared with other RefPointers. That is, the data encapsulated by a
   RefPointer is guaranteed not to be modified through side effects.

   Note that pointers returned by the various begin() and end() functions are
   not reference-counted, so you should not store the returned pointers.

   \ingroup Internals
*/
template <class value_type>
class RefPointer {
 public:
  typedef value_type elt_t;  ///< Type of data pointed to

  /** Create an empty reference */
  RefPointer();
  /** Allocate a pointer of s bytes. */
  RefPointer(size_t new_size);
  /** Wrap around the given data */
  RefPointer(elt_t *data, size_t size, bool owned = true);
  /** Copy constructor that increases the reference count. */
  RefPointer(const RefPointer<elt_t> &p);

  /** Destructor that deletes no longer reference data. */
  ~RefPointer();

  /** Copy a pointer increasing the reference count. */
  RefPointer<elt_t> &operator=(const RefPointer<elt_t> &p);

  /** Retreive the pointer without caring for references (unsafe). */
  elt_t *begin() {
    appropriate();
    return ref_->begin();
  }
  /** Retreive the pointer without caring for references (unsafe). */
  const elt_t *begin() const { return ref_->begin(); }
  /** Retreive the pointer without caring for references (unsafe). */
  const elt_t *begin_const() const { return ref_->begin(); }
  /** Retreive the pointer without caring for references (unsafe). */
  const elt_t *end_const() const { return ref_->end(); }
  /** Retreive the pointer without caring for references (unsafe). */
  const elt_t *end() const { return ref_->end(); }
  /** Retreive the pointer without caring for references (unsafe). */
  elt_t *end() {
    appropriate();
    return ref_->end();
  }

  /** Size of pointed-to data. */
  size_t size() const { return ref_->size(); }

  /** Number of references to the internal data */
  size_t ref_count() const { return ref_->references(); }

  /** Replace the pointer with newly allocated data. */
  void reallocate(size_t new_size);

 private:
  class pointer;
  mutable pointer *ref_;  // Pointer to data we reference or NULL

  /** Ensure that we have a unique copy of the data. If the pointer has more
      than one reference, a fresh new copy of the data is created.
  */
  void appropriate();
  pointer *reference() const;
  void dereference();
};

};  // namespace tensor

#include <tensor/detail/refcount.hpp>

#endif /* !REFCOUNT_REFCOUNT_H */
