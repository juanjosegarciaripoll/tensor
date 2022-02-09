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

#include <cassert>
#include <stdexcept>
#include <tensor/indices.h>
#include <tensor/io.h>

namespace tensor {

Dimensions dimensions_from_ranges(SimpleVector<Range> &ranges,
                                  const Dimensions &parent_dimensions) {
  SimpleVector<index> output(ranges.size());
  if (ranges.ssize() != parent_dimensions.rank()) {
    if (ranges.size() == 1) {
      // If we only provide one range, it is as if we were flattening the tensor
      // prior to iteration.
      return dimensions_from_ranges(ranges,
                                    Dimensions{parent_dimensions.total_size()});
    }
    throw std::out_of_range("Number of _ exceeds Tensor rank.");
  }
  for (index i = 0; i < ranges.ssize(); ++i) {
    Range &r = ranges.at(i);
    r.set_dimension(parent_dimensions[i]);
    output.at(i) = r.size();
  }
  return Dimensions(output);
}

static bool equispaced(const Indices &indices) {
  auto it = indices.begin();
  index first = *(it++);
  index last = *(it++);
  index delta = last - first;
  for (auto end = indices.end(); it != end; ++it) {
    index n = *it;
    if (n != last + delta) {
      return false;
    }
    last = n;
  }
  return true;
}

Range::Range(Indices indices)
    : start_{0},
      step_{1},
      limit_{static_cast<index>(indices.size())},
      dimension_{0},
      indices_(std::move(indices)) {
  if (indices.size() == 0) {
    *this = empty();
  } else if (indices.size() == 1) {
    *this = Range(indices[0]);
  } else if (equispaced(indices)) {
    index first = indices[0];
    index last = indices[indices.ssize() - 1];
    *this = Range(first, last, indices[1] - first);
  }
}

Range Range::empty() { return Range(0, -1, 1); }

const Range _ = Range::full();

Range Range::full(index start, index step) { return Range(start, -2, step); }

bool Range::maybe_combine(const Range &other) {
  /* Sometimes we can merge two range specifications into a simpler one. This
  can only be done when building iterators, not when building views, because it
  breaks the number of dimensions. */
  index total_range_size = dimension_ * other.dimension();
  // An empty range will combine with everything and save us further scans
  if (size() == 0 || other.size() == 0) {
    *this = Range::empty();
    dimension_ = total_range_size;
    return true;
  }
  if (!has_indices() && !other.has_indices()) {
    if (size() == 1) {
      step_ = other.step() * dimension_;
      limit_ = start_ + other.limit() * dimension_;
      start_ = start_ + other.start() * dimension_;
      dimension_ = total_range_size;
      return true;
    }
    if (other.size() == 1) {
      index offset = other.start() * dimension_;
      start_ += offset;
      limit_ += offset;
      dimension_ = total_range_size;
      return true;
    }
    if (is_full() && other.is_full()) {
      start_ = 0;
      limit_ = dimension_ = total_range_size;
      step_ = 1;
      return true;
    }
  }
  return false;
}

void Range::set_dimension(index dimension) {
  /* The description of a range is incomplete until we set its dimension, which
  can only be done when it references a tensor. */
  if (dimension >= limit()) {
    if (limit() < 0) {
      // When limit() < 0 we had a full range whose size must be updated
      limit_ = dimension;
    } else if (has_indices()) {
      // Verify that all positions are within range for the selected indices
      for (auto x : indices()) {
        if (x >= dimension) {
          throw std::out_of_range("Range indices exceed tensor dimensions");
        }
      }
    }
    dimension_ = dimension;
  } else {
    // The range [start,end] falls outside [0,dimension)
    throw std::out_of_range("Range indices exceed tensor dimensions");
  }
}

const Range RangeIterator::empty_range = Range::empty();

RangeIterator RangeIterator::make_range_iterators(
    const SimpleVector<Range> &ranges, end_flag_t end_flag) {
  if (ranges.size() == 0 ||
      std::any_of(ranges.begin(), ranges.end(),
                  [](const Range &s) { return s.size() == 0; })) {
    return RangeIterator(Range::empty(), 1, end_flag);
  } else {
    return make_next_iterator(ranges.begin(), ranges.ssize(), 1, end_flag);
  }
}

RangeIterator RangeIterator::make_next_iterator(const Range *ranges,
                                                index ranges_left, index factor,
                                                end_flag_t end_flag) {
  Range r = *ranges;
  ++ranges;
  --ranges_left;
  while (ranges_left && r.maybe_combine(*ranges)) {
    --ranges_left;
    ++ranges;
  }
  RangeIterator *next = nullptr;
  if (ranges_left) {
    next = new RangeIterator(make_next_iterator(
        ranges, ranges_left, factor * r.dimension(), end_flag));
  }
  return RangeIterator(r, factor, end_flag, next_t(next));
}

RangeIterator::RangeIterator(const Range &r, index factor, end_flag_t end_flag,
                             next_t next)
    : offset_{next ? next->get_position() : 0},
      indices_(r.indices()),
      next_{next} {
  /*
   * When we work with indices, the current position is
   *    step_ * v[counter_] + offset_
   * and counter_ grows from 0 up to limit_.
   * 
   * When we work with bare indices, the current position is counter_,
   * which starts at offset_, is incremented by step_ and is always
   * below limit_
   */
  factor_ = factor;
  if (r.has_indices()) {
    start_ = 0;
    limit_ = r.indices().ssize();
    step_ = 1;
  } else {
    step_ = r.step() * factor;
    limit_ = r.limit() * factor;
    start_ = r.start() * factor;
  }
  if (end_flag == range_end) {
    counter_ = limit_;
  } else {
    counter_ = start_;
  }
}

index RangeIterator::get_position() const {
  index n = counter_;
  if (has_indices()) {
    n = indices()[n] * factor_;
  }
  return n + offset_;
}

void RangeIterator::advance_next() {
  if (next_) {
    ++(*next_);
    offset_ = next_->get_position();
    if (!next_->finished()) {
      counter_ = start_;
      return;
    }
  }
  counter_ = limit_;
}

}  // namespace tensor
