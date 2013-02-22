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

#ifndef TENSOR_REFCOUNT_H
#define TENSOR_REFCOUNT_H

#include <cstring>
#include <algorithm>

namespace tensor {

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
  RefPointer();
  /** Allocate a pointer of s bytes. */
  RefPointer(size_t new_size);
  /** Copy constructor that increases the reference count. */
  RefPointer(const RefPointer<elt_t> &p);
  /** Reference count a given data */
  RefPointer(elt_t *data, size_t size);

  /** Destructor that deletes no longer reference data. */
  ~RefPointer();

  /** Copy a pointer increasing the reference count. */
  RefPointer<elt_t> &operator=(const RefPointer<elt_t> &p);

  /** Retreive the pointer without caring for references (unsafe). */
  elt_t *begin() { appropiate(); return ref_->begin(); }
  /** Retreive the pointer without caring for references (unsafe). */
  const elt_t *begin() const { return ref_->begin(); }
  /** Retreive the pointer without caring for references (unsafe). */
  const elt_t *begin_const() const { return ref_->begin(); }
  /** Retreive the pointer without caring for references (unsafe). */
  const elt_t *end_const() const { return ref_->end(); }
  /** Retreive the pointer without caring for references (unsafe). */
  const elt_t *end() const { return ref_->end(); }
  /** Retreive the pointer without caring for references (unsafe). */
  elt_t *end() { appropiate(); return ref_->end(); }

  /** Size of pointed-to data. */
  size_t size() const { return ref_->size(); }

  /** Reference counter */
  size_t ref_count() const { return ref_->references(); }
  /** Reference counter */
  bool other_references() const { return ref_->references() > 1; }
  /** Ensure that we have a unique copy of the data. If the pointer has more
      than one reference, a fresh new copy of the data is created.
  */
  void appropiate();

  /** Replace the pointer with newly allocated data. */
  void reallocate(size_t new_size);
  /** Set to NULL */
  void reset() { reset(0, 0); }
  /** Set to some pointer */
  void reset(elt_t *p, size_t new_size = 1);

private:
  class pointer;

  mutable pointer *ref_; // Pointer to data we reference or NULL
};

}; // namespace

#include <tensor/detail/refcount.hpp>

#endif /* !REFCOUNT_REFCOUNT_H */
