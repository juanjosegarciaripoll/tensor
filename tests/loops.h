//
// Copyright 2008, Juan Jose Garcia-Ripoll
//
#pragma once

#ifndef TENSOR_TEST_LOOPS_H
#define TENSOR_TEST_LOOPS_H

#include <algorithm>
#include <iostream>
#include <string>
#include <gtest/gtest.h>
#ifndef GTEST_INCLUDE_GTEST_GTEST_DEATH_TEST_H_
#include <gtest/gtest-death-test.h>
#endif
#include <tensor/rand.h>
#include <tensor/tensor.h>
#include <tensor/io.h>
#include <tensor/tools.h>

#define EPSILON 1e-14

#ifdef TENSOR_DEBUG
#define ONLY_IN_DEBUG(x) (x)
#define ASSERT_THROW_DEBUG(x, y) ASSERT_THROW(x, y)
#else
#define ONLY_IN_DEBUG(x)
#define ASSERT_THROW_DEBUG(x, y) ASSERT_DEATH(x, ".*")
#endif

namespace tensor_test {

using namespace tensor;

static constexpr double pi = 3.14159265358979323846;

/*
   * Verifies that the tensor that has been passed to the routine has
   * not been reallocated.
   */
template <class Tensor>
void unchanged(const Tensor &t1, const Tensor &t2, size_t expected_refs = 2) {
  if ((t1.size() + t2.size()) == 0) return;
  EXPECT_EQ(t1.cbegin(), t2.cbegin());
  EXPECT_EQ(expected_refs, t1.ref_count());
  EXPECT_EQ(expected_refs, t2.ref_count());
}

/*
   * Verifies that the tensor that has been passed to the routine has
   * not been reallocated.
   */
template <class Tensor>
void unique(const Tensor &t) {
  if (t.size()) {
    EXPECT_EQ(1, t.ref_count());
  }
}

/*
   * Approximately equal numbers.
   */
template <typename elt_t1, typename elt_t2>
bool simeq(elt_t1 a, elt_t2 b, double epsilon = 2 * EPSILON) {
  double x = tensor::abs(a - b);
  if (x > epsilon) {
    std::cout << x << std::endl;
    return false;
  }
  return true;
}

template <typename elt_t>
bool simeq(const Tensor<elt_t> &a, const Tensor<elt_t> &b,
           double epsilon = 2 * EPSILON) {
  if (a.rank() != b.rank()) {
    std::cerr << "Comparing tensors of different ranks: " << a.rank() << " vs "
              << b.rank() << '\n';
    return false;
  }

  if (!all_equal(a.dimensions(), b.dimensions())) {
    std::cerr << "Dimensions do not match:" << a.dimensions() << " vs. "
              << b.dimensions() << '\n';
    return false;
  }

  for (typename Tensor<elt_t>::const_iterator ia = a.begin(), ib = b.begin();
       ia != a.end(); ++ia, ++ib) {
    double x = tensor::abs(*ia - *ib);
    if (x > epsilon) {
      std::cout << x << std::endl;
      return false;
    }
  }
  return true;
}

#define EXPECT_CEQ(a, b) EXPECT_TRUE(simeq(a, b))
#define EXPECT_CEQ3(a, b, c) EXPECT_TRUE(simeq(a, b, c))
#define ASSERT_CEQ(a, b) ASSERT_TRUE(simeq(a, b))

/*
   * Approximately equal tensors.
   */
template <class Tensor>
bool approx_eq(const Tensor &A, const Tensor &B, double epsilon = 2 * EPSILON) {
  if (A.rank() != B.rank()) {
    std::cout << "Ranks do not match." << std::endl;
    return false;
  }

  if (!all_equal(A.dimensions(), B.dimensions())) {
    std::cout << "Dimensions do not match." << std::endl
              << A.dimensions() << " vs. " << B.dimensions() << std::endl;
    return false;
  }

  double x = norm0(A - B);
  if (x > epsilon) {
    std::cout << "Deviation: " << x << std::endl;
    return false;
  }

  return true;
}

template <typename elt_t>
bool unitaryp(const Tensor<elt_t> &U, double epsilon = EPSILON) {
  Tensor<elt_t> Ut = adjoint(U);
  if (U.rows() <= U.columns()) {
    if (!approx_eq(mmult(U, Ut), Tensor<elt_t>::eye(U.rows()), epsilon))
      return false;
  }
  if (U.columns() <= U.rows()) {
    if (!approx_eq(mmult(Ut, U), Tensor<elt_t>::eye(U.columns()), epsilon))
      return false;
  }
  return true;
}

/*
   * Test over integers.
   */
inline void test_over_integers(int min, int max, void test(int)) {
  for (; min <= max; ++min) {
    test(min);
  }
}

/*
   * Creates a vector of random dimensions.
   */
inline Indices random_dimensions(int rank, int max_dim) {
  Indices dims(rank);
  for (int i = 0; i < rank; ++i) {
    dims.at(i) = rand<int>(0, max_dim + 1);
  }
  return dims;
}

/*
   * Creates a vector of random dimensions some of which are empty
   */
inline Indices random_empty_dimensions(int rank, int max_dim, int which = -1) {
  Indices dims = random_dimensions(rank, max_dim);
  if (which < 0 || which >= rank) which = rand<int>(rank);
  dims.at(which) = 0;
  return dims;
}

/*
   * Loop over dimensions
   */

class DimensionIterator {
 public:
  explicit DimensionIterator(int rank, int max_dim = 10)
      : dims_(rank), max_(max_dim), more_(true) {
    std::fill(dims_.begin(), dims_.end(), 0);
  }

  bool operator++() {
    for (int i = 0; i < dims_.size(); ++i) {
      if (++dims_.at(i) < max_) {
        return more_ = true;
      }
      dims_.at(i) = 0;
    }
    return more_ = false;
  }

  const Indices &operator*() const { return dims_; }

  operator bool() const { return more_; }

 private:
  Indices dims_;
  int max_;
  bool more_;
};

class fixed_rank_iterator {
 public:
  explicit fixed_rank_iterator(int rank, int max_dimension = 10)
      : rank_(rank),
        max_dimension_(max_dimension),
        indices_(rank),
        finished_(false) {
    tensor_assert((rank >= 0) && (max_dimension >= 0));
    reset();
  }
  int rank() const { return rank_; }
  int max_dimension() const { return max_dimension_; }
  bool finished() const { return finished_; }
  bool more() const { return !finished_; }
  const Indices &dimensions() const { return indices_; }
  void reset() { std::fill(indices_.begin(), indices_.end(), 0); }
  bool next() {
    if (!finished_) {
      for (int i = 0; i < rank(); ++i) {
        if (++indices_.at(i) < max_dimension()) {
          return true;
        }
        indices_.at(i) = 0;
      }
      finished_ = true;
    }
    return false;
  }

 private:
  int rank_, max_dimension_;
  Indices indices_;
  bool finished_;
};

template <typename elt_t>
Tensor<elt_t> tensor_with_increasing_values(const Dimensions &d) {
  Tensor<elt_t> data(d);
  elt_t accum = 0;
  for (elt_t &x : data) {
    x = accum;
    accum += 1;
  }
  return data;
}

template <typename elt_t, typename unop>
void test_over_fixed_rank_tensors(unop test, int rank, int max_dimension = 10) {
  for (fixed_rank_iterator it(rank, max_dimension); !it.finished(); it.next()) {
    Tensor<elt_t> data = tensor_with_increasing_values<elt_t>(it.dimensions());
    test(data);
  }
}

template <typename elt_t, typename binop>
void test_over_fixed_rank_pairs(binop test, int rank, int max_dimension = 6) {
  for (fixed_rank_iterator it1(rank, max_dimension); !it1.finished();
       it1.next()) {
    Tensor<elt_t> data1 =
        tensor_with_increasing_values<elt_t>(it1.dimensions());
    for (fixed_rank_iterator it2(rank, max_dimension); !it2.finished();
         it2.next()) {
      Tensor<elt_t> data2 =
          tensor_with_increasing_values<elt_t>(it2.dimensions());
      test(data1, data2);
    }
  }
}

/*
   * Test over all tensor sizes and ranks, randomly.
   */
template <typename elt_t>
void test_over_all_tensors(void test(Tensor<elt_t> &t), int max_rank = 4,
                           int max_dimension = 10) {
  for (int rank = 0; rank <= max_rank; ++rank) {
    std::ostringstream rank_string;
    rank_string << "rank: " << rank;
    SCOPED_TRACE(rank_string.str());
    test_over_fixed_rank_tensors(test, rank, max_dimension);
  }
}

template <typename elt_t>
void test_over_tensors(void test(Tensor<elt_t> &t), int max_rank = 4,
                       int max_dimension = 10, int max_times = 15) {
  for (int rank = 0; rank <= max_rank; ++rank) {
    std::ostringstream rank_string;
    rank_string << "rank: " << rank;
    SCOPED_TRACE(rank_string.str());
    //
    // Test over random dimensions
    //
    for (int times = 0; times < max_times; ++times) {
      Tensor<elt_t> data(random_dimensions(rank, max_dimension));
      data.randomize();
      test(data);
    }
    //
    // Forced tests over empty tensors
    //
    for (int times = 0; times < rank; ++times) {
      Tensor<elt_t> data(random_empty_dimensions(rank, max_dimension, times));
      test(data);
    }
  }
}

class DimensionsProducer {
 public:
  explicit DimensionsProducer(const Indices &d)
      : base_indices(d), counter(13) {}

  operator bool() const { return counter >= 14; }
  int operator++() { return counter++; }

  Indices operator*() const {
    RTensor P;
    switch (counter) {
        // 1D Tensor<elt_t>
      case 1:
        P = RTensor::empty(d(0) * d(1) * d(2) * d(3));
        break;
        // 2D Tensor<elt_t>
      case 2:
        P = RTensor::empty(d(0), d(1) * d(2) * d(3));
        break;
      case 3:
        P = RTensor::empty(d(0) * d(1), d(2) * d(3));
        break;
      case 4:
        P = RTensor::empty(d(0) * d(1) * d(2), d(3));
        break;
      case 5:
        P = RTensor::empty(d(3), d(0) * d(1) * d(2));
        break;
      case 6:
        P = RTensor::empty(d(2) * d(3), d(0) * d(1));
        break;
        // 3D Tensor<elt_t>
      case 7:
        P = RTensor::empty(d(0), d(1), d(2) * d(3));
        break;
      case 8:
        P = RTensor::empty(d(0) * d(1), d(2), d(3));
        break;
      case 9:
        P = RTensor::empty(d(3) * d(2), d(0), d(1));
        break;
        // 4D Tensor<elt_t>
      case 10:
        P = RTensor::empty(d(0), d(1), d(2), d(3));
        break;
      case 11:
        P = RTensor::empty(d(3), d(1), d(2), d(0));
        break;
      case 12:
        P = RTensor::empty(d(1), d(0), d(3), d(2));
        break;
      case 13:
        P = RTensor::empty(d(2), d(0), d(3), d(1));
        break;
    }
    return P.dimensions();
  }

 private:
  Indices::elt_t d(int which) const {
    if (which >= base_indices.size()) {
      return 1;
    } else {
      return base_indices[which];
    }
  }

  Indices base_indices;
  int counter;
};

template <typename elt_t>
Tensor<elt_t> random_unitary(int n, int iterations = -1);
template <>
RTensor random_unitary(int n, int iterations);
template <>
CTensor random_unitary(int n, int iterations);
RTensor random_permutation(int n, int iterations = -1);

static struct Foo {
  Foo() { ::tensor::tensor_abort_handler(); }
} foo;

// Iterates through all true/false combinations of a Booleans of given rank.
class BooleansIterator {
 public:
  explicit BooleansIterator(int rank) : base_booleans_(rank), more_(true) {
    std::fill(base_booleans_.begin(), base_booleans_.end(), false);
  }

  bool operator++() {
    for (int i = 0; i < base_booleans_.size(); ++i) {
      if (base_booleans_[i] == false) {
        base_booleans_.at(i) = true;
        return more_ = true;
      }

      base_booleans_.at(i) = false;
    }

    return more_ = false;
  }

  operator bool() const { return more_; }

  const Booleans &operator*() const { return base_booleans_; }

 private:
  Booleans base_booleans_;
  bool more_;
};

//
// FIXTURES
//

template <typename T>
class TensorTest : public ::testing::Test {
 public:
  using value_type = typename T::elt_t;

  value_type small_number() const {
    static constexpr double small_factor = 1e-6;
    return small_factor * rand<value_type>();
  }

  constexpr value_type one() const { return number_one<value_type>(); }

  template <typename otherT>
  constexpr value_type to_value_type(otherT x) {
    return static_cast<value_type>(x);
  }
};

}  // namespace tensor_test

#endif /* !TENSOR_TEST_LOOPS_H */
