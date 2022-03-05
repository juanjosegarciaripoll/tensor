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

#include <stdexcept>
#include <algorithm>
#include <functional>
#include <tensor/exceptions.h>
#include <tensor/indices.h>
#include <tensor/io.h>

namespace tensor {

Dimensions squeeze_dimensions(const Dimensions &dims) {
  index ones = std::count(std::begin(dims), std::end(dims), 1);
  if (ones == dims.rank()) {
    // A(range(0), range(3), range(1)) has dimensions [1,1,1] which
    // squeeze down to [1].
    return Dimensions{1};
  } else if (ones > 0) {
    SimpleVector<index> without_ones(dims.rank() - ones);
    std::copy_if(std::begin(dims), std::end(dims), std::begin(without_ones),
                 [](index n) { return n != 1; });
    return Dimensions(without_ones);
  }
  return dims;
}

Range RangeSpan::next_range() {
  Range r = *begin_;
  ++begin_;
  while (begin_ != end_ && r.maybe_combine(*begin_)) {
    ++begin_;
  }
  return r;
}

bool RangeSpan::empty_ranges() const {
  auto nonempty_range = [](const Range &range) { return range.size() != 0; };
  return std::none_of(begin_, end_, nonempty_range);
}

bool RangeSpan::valid_ranges() const {
  auto valid_range = [](const Range &range) { return range.dimension() >= 0; };
  return std::all_of(begin_, end_, valid_range);
}

Dimensions RangeSpan::get_dimensions(const Dimensions &parent_dimensions) {
  if (ssize() != parent_dimensions.rank()) {
    if (ssize() == 1) {
      // If we only provide one range, it is as if we were flattening the tensor
      // prior to iteration.
      return get_dimensions(Dimensions{parent_dimensions.total_size()});
    }
    throw std::out_of_range("Number of _ exceeds Tensor rank.");
  }
#ifdef TENSOR_RANGE_SQUEEZE
  index removed_dimensions = std::count_if(
      begin(), end(), [](const Range &r) { return r.squeezed(); });
  index final_size = ssize() - removed_dimensions;
  index total_size = 1;
  SimpleVector<index> output(final_size);
  for (index i = 0, j = 0; i < ssize(); ++i) {
    Range &r = at(i);
    r.set_dimension(parent_dimensions[i]);
    if (!r.squeezed()) {
      index s = r.size();
      total_size *= s;
      output.at(j++) = s;
    }
  }
  if (final_size == 0) {
    return Dimensions{total_size};
  }
#else
#error "TENSOR_RANGE_SQUEEZE is outdated!"
  SimpleVector<index> output(size());
  for (index i = 0, j = 0; i < ssize(); ++i) {
    Range &r = at(i);
    r.set_dimension(parent_dimensions[i]);
    output.at(j++) = r.size();
  }
#endif
  return output;
}

static bool equispaced(const Indices &indices) {
  auto it = indices.begin();
  index first = *(it++);
  index last = *(it++);
  index offset = last - first;
  for (auto end = indices.end(); it != end; ++it) {
    index n = *it;
    if (n != last + offset) {
      return false;
    }
    last = n;
  }
  return true;
}

Range::Range(Indices indices)
    : first_{0},
      step_{1},
      last_{indices.ssize() - 1},
      dimension_{-1},
      indices_(std::move(indices)) {
  if (indices.size() == 0) {
    *this = empty();
  } else if (indices.size() == 1) {
    *this = Range(indices[0], indices[0]);
  } else if (equispaced(indices)) {
    index first = indices[0];
    index last = indices[indices.ssize() - 1];
    *this = Range(first, last, indices[1] - first);
  }
}

/* Because we allow for negative indices, there are not many ways we can express
   empty intervals. For instance, (first,end)=(0,-1) would get reinterpreted as
   (0, dimension-1). This particular choice always produces, after
   set_dimension() is called, an empty interval which is within bounds. */
Range Range::empty() { return Range(-1, -2, 1); }

Range Range::empty(index dimension) {
  Range output(0, -1, 1);
  output.dimension_ = dimension;
  return output;
}

const Range _ = Range::full();

Range Range::full(index start, index step) { return Range(start, -1, step); }

bool Range::maybe_combine(const Range &other) {
  /* Sometimes we can merge two range specifications into a simpler one. This
  can only be done when building iterators, not when building views, because it
  breaks the number of dimensions. */
  index total_range_size = dimension_ * other.dimension();
  // An empty range will combine with everything and save us further scans
  if (size() == 0 || other.size() == 0) {
    *this = Range::empty(total_range_size);
    return true;
  }
  if (!has_indices() && !other.has_indices()) {
    if (size() == 1) {
      // This range only defines a global displacement, determined by first()
      index offset = first_;
      step_ = other.step() * dimension_;
      last_ = offset + other.last() * dimension_;
      first_ = offset + other.first() * dimension_;
      dimension_ = total_range_size;
      return true;
    }
    if (other.size() == 1) {
      // The other range is now our offset.
      index offset = other.first() * dimension_;
      first_ += offset;
      last_ += offset;
      dimension_ = total_range_size;
      return true;
    }
    if (is_full() && other.is_full()) {
      first_ = 0;
      dimension_ = total_range_size;
      last_ = total_range_size - 1;
      step_ = 1;
      return true;
    }
  }
  return false;
}

void Range::set_dimension(index new_dimension) {
  /* The description of a range is incomplete until we set its dimension, which
  can only be done when it references a tensor. */
  if (new_dimension < 0) {
    throw invalid_dimension();
  } else if (new_dimension == 0) {
    if (has_indices() || (first_ != 0 && first_ != -1)) {
      throw out_of_bounds_index();
    } else {
      *this = Range::empty(0);
      return;
    }
  } else if (dimension_ >= 0 && new_dimension != dimension_) {
    throw std::invalid_argument("Cannot reset dimension of a range");
  } else if (has_indices()) {
    // Verify that all positions are within range for the selected indices
    for (auto x : indices()) {
      x = Dimensions::normalize_index_safe(x, new_dimension);
    }
  } else {
    /* We reinterpret start_ and last_ when they have negative values
     * as relative to one past the last element (i.e. -1 = last element).
     * We then make the intersection between [start_,last_] and
     * [0, new_dimension].
     */
    if (first_ < 0) {
      first_ = first_ + new_dimension;
    }
    if (first_ < 0 || first_ >= new_dimension) {
      throw out_of_bounds_index();
    }
    if (last_ < 0) {
      last_ = last_ + new_dimension;
    }
    last_ = std::max<index>(-1, std::min<index>(last_, new_dimension - 1));
    index delta = last_ - first_;
    if (delta * step_ < 0) {
      // When the range limits (first, last) have a direction
      // opposite to step, the range is empty.
      *this = Range::empty(new_dimension);
      return;
    }
    last_ = first_ + step_ * (delta / step_);
  }
  dimension_ = new_dimension;
}

const Range RangeIterator::empty_range = Range::empty();

bool RangeIterator::same_iterator(const RangeIterator &r) const {
  if ((counter_ != r.counter_) or (limit_ != r.limit_) or
      (start_ != r.start_) or (step_ != r.step_) or (factor_ != r.factor_) or
      (has_indices() != r.has_indices()) or (bool(next_) != bool(r.next_)))
    return false;
  if (has_indices() && !all_equal(indices_, r.indices_)) return false;
  if (next_) {
    return next_->same_iterator(*r.next_);
  }
  return true;
}

RangeIterator RangeIterator::make_end_iterator() const {
  RangeIterator output;
  output.counter_ = output.limit_ = limit_;
  return output;
}

RangeIterator RangeIterator::make_range_iterators(RangeSpan &ranges,
                                                  end_flag_t end_flag) {
  if (ranges.empty_ranges()) {
    return RangeIterator(Range::empty(0), 1, end_flag);
  }
  tensor_assert2(
      ranges.valid_ranges(),
      std::invalid_argument(
          "Invalid dimension size in Range found when creating RangeIterator"));
  return make_next_iterator(ranges, 1, end_flag);
}

RangeIterator::RangeIterator(const RangeIterator &r)
    : counter_{r.counter_},
      limit_{r.limit_},
      step_{r.step_},
      offset_{r.offset_},
      factor_{r.factor_},
      start_{r.start_},
      indices_(r.indices_),
      next_{r.next_ ? new RangeIterator(*r.next_) : nullptr} {}

RangeIterator RangeIterator::make_next_iterator(RangeSpan &ranges, index factor,
                                                end_flag_t end_flag) {
  Range r = ranges.next_range();
  RangeIterator *next = nullptr;
  if (ranges.more() && end_flag != range_end) {
    next = new RangeIterator(
        make_next_iterator(ranges, factor * r.dimension(), end_flag));
  }
  return RangeIterator(r, factor, end_flag, next_t(next));
}

RangeIterator::RangeIterator(const Range &r, index factor, end_flag_t end_flag,
                             next_t next)
    : offset_{next ? next->get_position() : 0},
      factor_{factor},
      indices_(r.indices()),
      next_{next} {
  /*
   * When we work with indices, the current position is
   *    step_ * v[counter_] + offset_
   * and counter_ grows from 0 up to limit_.
   * 
   * When we work with bare indices, the current position is counter_,
   * which firsts at offset_, is incremented by step_ and is always
   * below limit_
   */
  if (r.dimension() < 0) {
    throw std::invalid_argument(
        "RangeIterator passed a Range without dimensions");
  }
  if (r.has_indices()) {
    start_ = 0;
    step_ = 0;
    limit_ = r.indices().ssize();
  } else {
    start_ = r.first() * factor;
    step_ = r.step() * factor;
    limit_ = r.size();
    offset_ += start_;
  }
  if (end_flag == range_end) {
    counter_ = limit_;
  } else {
    counter_ = 0;
  }
}

index RangeIterator::get_position() const {
  if (has_indices()) {
    return offset_ + factor_ * indices()[counter_];
  } else {
    return offset_;
  }
}

void RangeIterator::advance_next() {
  if (next_) {
    ++(*next_);
    if (!next_->finished()) {
      offset_ = next_->get_position() + start_;
      counter_ = 0;
      return;
    }
  }
  counter_ = limit_;
}

}  // namespace tensor
