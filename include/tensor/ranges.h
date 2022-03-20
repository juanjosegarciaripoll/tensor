#pragma once
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

#ifndef TENSOR_RANGES_H
#define TENSOR_RANGES_H

#include <tensor/indices.h>

/*!\addtogroup Tensors */
/*@{*/
namespace tensor {

//////////////////////////////////////////////////////////////////////
// RANGE OF INTEGERS
//

/** Range of indices. This class should never be used by public functions, but
    only as the output of the function range and only to access segments of
    a tensors, as in
    \code
    b = a(range(1,2),_)
    \endcode
  */
class Range {
  /* The logic of ranges is that we define the start, the step and one past the
     last element we visit (the limit). We combine this with the wraparound
     semantics from Numpy, by which negative numbers are associated positions
     relative to the dimension of the coordinate we are running through. Thus,

     (first, last) = (0, 0)   -> runs over [0]
                   = (0, -1)  -> runs over [0, 1, 2] for dimension = 3
                   = (-2, -1) -> runs over [1, 2] for dimension = 3

     Since we can store negative numbers, the `limit=end+1` can also take
     negative or zero values.
   */
 public:
  explicit Range(index position) : Range(position, position, 1) {
    squeezed_ = true;
  }
  Range(index first, index last) : Range(first, last, 1) {}
  Range(index first, index last, index step)
      : first_{first}, step_{step}, last_{last}, dimension_{-1} {
    tensor_assert2(step != 0, std::invalid_argument("Range() with zero step"));
  }
  Range(index first, index last, index step, index dimension)
      : first_{first}, step_{step}, last_{last}, dimension_{-1} {
    tensor_assert2(step != 0, std::invalid_argument("Range() with zero step"));
    set_dimension(dimension);
  }
  // NOLINTNEXTLINE(*-explicit-constructor)
  // cppcheck-suppress noExplicitConstructor
  Range(Indices indices);
  Range() = default;
  Range(const Range &r) = default;
  Range(Range &&r) = default;
  Range &operator=(const Range &r) = default;
  Range &operator=(Range &&r) = default;

  index first() const { return first_; }
  index step() const { return step_; }
  index last() const { return last_; }
  index dimension() const { return dimension_; }
  void set_dimension(index new_dimension);
  bool has_indices() const { return indices().size() != 0; }
  const Indices &indices() const { return indices_; }
  index get_index(index pos) const { return indices()[pos]; }
  index size() const {
    return std::max<index>(0, 1 + (last_ - first_) / step_);
  }
  index is_full() const {
    return first_ == 0 && step_ == 1 &&
           (last_ == -1 || last_ == dimension_ - 1);
  }
  bool squeezed() const { return squeezed_; }
  bool maybe_combine(const Range &other);

  static Range empty();
  static Range empty(index dimension);
  static Range full(index start = 0, index step = 1);

 private:
  index first_{-1}, step_{1}, last_{-2}, dimension_{-1};
  Indices indices_;
  bool squeezed_{false};
  bool dimension_undefined() const { return dimension_ < 0; }
};

extern const Range _;

class RangeSpan {
  Range *begin_;
  Range *end_;

 public:
  template <size_t N>
  explicit RangeSpan(std::array<Range, N> &v)
      : begin_{&v[0]}, end_{begin_ + N} {}
  explicit RangeSpan(SimpleVector<Range> &v)
      : begin_{v.begin()}, end_{v.end()} {}
  bool more() const { return begin_ != end_; }
  bool empty_ranges() const;
  bool valid_ranges() const;
  Range next_range();
  Range &at(index i) { return begin_[i]; }
  const Range *begin() const { return begin_; }
  const Range *end() const { return end_; }
  index ssize() const { return end_ - begin_; }
  Dimensions get_dimensions(const Dimensions &parent_dimensions);
};

class RangeIterator {
 public:
  static const Range empty_range;
  typedef std::unique_ptr<RangeIterator> next_t;

  RangeIterator() = default;
  RangeIterator(RangeIterator &&r) = default;
  RangeIterator(const RangeIterator &r);
  RangeIterator &operator=(const RangeIterator &r) = delete;
  RangeIterator &operator=(RangeIterator &&r) = default;

  index operator*() const { return get_position(); };
  RangeIterator &operator++() {
    if (++counter_ >= limit_) {
      advance_next();
    } else {
      offset_ += step_;
    }
    return *this;
  }
  bool finished() const { return counter_ >= limit_; }
  bool operator!=(const RangeIterator &other) const {
    return other.counter_ != counter_;
  }
  bool operator==(const RangeIterator &other) const {
    return other.counter_ == counter_;
  }
  static RangeIterator begin(RangeSpan &ranges) {
    return make_range_iterators(ranges);
  }
  static RangeIterator end(RangeSpan &ranges) {
    return begin(ranges).make_end_iterator();
  }
  static RangeIterator begin(SimpleVector<Range> ranges) {
    auto span = RangeSpan(ranges);
    return begin(span);
  }
  static RangeIterator end(SimpleVector<Range> ranges) {
    auto span = RangeSpan(ranges);
    return begin(span).make_end_iterator();
  }
  RangeIterator make_end_iterator() const noexcept;
  index get_position() const noexcept {
    if (has_indices()) {
      return offset_ + factor_ * indices()[counter_];
    } else {
      return offset_;
    }
  };
  index counter() const noexcept { return counter_; }
  index offset() const noexcept { return offset_; }
  index limit() const noexcept { return limit_; }
  index step() const noexcept { return step_; }
  bool has_next() const noexcept { return next_ != nullptr; }
  const RangeIterator &next() const noexcept { return *next_; }
  void advance_next() noexcept;
  bool has_indices() const noexcept { return indices_.size() != 0; }
  const Indices &indices() const noexcept { return indices_; }
  bool same_iterator(const RangeIterator &it) const noexcept;
  bool contiguous() const noexcept { return !has_indices() && step_ == 1; }

 private:
  index counter_{}, limit_{}, step_{}, offset_{}, factor_{}, start_{};
  Indices indices_{};
  next_t next_{};

  RangeIterator(const Range &r, index factor = 1,
                RangeIterator *next = nullptr);
  static RangeIterator make_range_iterators(RangeSpan &ranges);

  static RangeIterator make_next_iterator(RangeSpan &ranges, index factor);
};

/**Create a Range which only contains one index. \sa \ref sec_tensor_view*/
inline Range range(index ndx) { return Range(ndx); }
/**Create a Range start:end (Matlab notation). \sa \ref sec_tensor_view*/
inline Range range(index start, index end) { return Range(start, end); }
/**Create a Range start:step:end (Matlab notation). \sa \ref sec_tensor_view*/
inline Range range(index start, index end, index step) {
  return Range(start, end, step);
}
/**Create a Range with the give set of indices. \sa \ref sec_tensor_view*/
inline Range range(Indices i) { return Range(i); }
/**Create a Range which covers all indices. \ref sec_tensor_view*/
inline Range range() { return Range::full(); }

}  // namespace tensor

#endif  // TENSOR_RANGES_H
