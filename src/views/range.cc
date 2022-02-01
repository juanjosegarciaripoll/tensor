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

namespace tensor {

Dimensions dimensions_from_ranges(SimpleVector<Range> &ranges,
                                  const Dimensions &parent_dimensions) {
  SimpleVector<index> output(ranges.size());
  if (ranges.size() != parent_dimensions.rank()) {
    throw std::out_of_range("Number of range() exceeds Tensor rank.");
  }
  for (index i = 0; i < ranges.size(); ++i) {
    Range &r = ranges.at(i);
    r.set_dimension(parent_dimensions[i]);
    output.at(i) = r.size();
  }
  return Dimensions(output);
}

Range Range::empty() { return Range(0, -1, 1); }

Range Range::full(index start, index step) { return Range(start, -2, step); }

void Range::set_dimension(index dimension) {
  if (dimension >= limit()) {
    dimension_ = dimension;
    if (limit() < 0) {
      // When limit() < 0 we had a full range whose size must be updated
      limit_ = dimension_;
    } else if (has_indices()) {
      // Verify that all positions are within range for the selected indices
      for (auto x : *indices_) {
        if (x >= dimension_) {
          throw std::out_of_range("Range indices exceed tensor dimensions");
        }
      }
    }
  } else {
    // The range [start,end] falls outside [0,dimension)
    throw std::out_of_range("Range indices exceed tensor dimensions");
  }
}

const Range RangeIterator::empty_range = Range::empty();

RangeIterator::RangeIterator(const Range *ranges, index factor, index left,
                             end_flag_t end_flag)
    : range_(left >= 1 ? ranges[0] : empty_range),
      counter_{end_flag == range_end ? range_.limit() : range_.start()},
      factor_{factor},
      offset_{0},
      next_{} {
  // If this Range is empty, we can stop here.
  if (left > 1 && range_.size()) {
    next_ = next_t(new RangeIterator(ranges + 1, factor * ranges[0].dimension(),
                                     left - 1, end_flag));
    offset_ = next_->get_position();
    if (next_->finished()) {
      counter_ = range_.limit();
      next_ = nullptr;
    }
  }
}

RangeIterator::RangeIterator(const Range &r, index factor)
    : range_{r}, counter_{r.start()}, factor_{factor}, offset_{0}, next_{} {}

index RangeIterator::get_position() const {
  index output = counter_;
  if (range_.has_indices()) {
    if (output >= range_.limit()) {
      output = -1;
    } else {
      output = range_.get_index(output);
    }
  }
  return output * factor_ + offset_;
}

RangeIterator &RangeIterator::operator++() {
  if ((counter_ += range_.step()) >= range_.limit()) {
    if (next_) {
      ++(*next_);
      offset_ = next_->get_position();
      if (!next_->finished()) {
        counter_ = range_.start();
        return *this;
      }
    }
    counter_ = range_.limit();
  }
  return *this;
}

}  // namespace tensor
