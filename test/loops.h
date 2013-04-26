// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_TEST_LOOPS_H
#define TENSOR_TEST_LOOPS_H

#include <algorithm>
#include <iostream>
#include <gtest/gtest.h>
#include <tensor/rand.h>
#include <tensor/tensor.h>
#include <tensor/io.h>
#include <tensor/tools.h>

#define EPSILON 1e-14

#ifdef NDEBUG
#define ONLY_IN_DEBUG(x)
#else
#define ONLY_IN_DEBUG(x) x
#endif

namespace tensor_test {

  using namespace tensor;

  /*
   * Verifies that the tensor that has been passed to the routine has
   * not been reallocated.
   */
  template<class Tensor>
  void unchanged(const Tensor &t1, const Tensor &t2, size_t expected_refs = 2) {
    if ((t1.size() + t2.size()) == 0)
      return;
    EXPECT_EQ(t1.begin_const(), t2.begin_const());
    EXPECT_EQ(expected_refs, t1.ref_count());
    EXPECT_EQ(expected_refs, t2.ref_count());
  }

  /*
   * Verifies that the tensor that has been passed to the routine has
   * not been reallocated.
   */
  template<class Tensor>
  void unique(const Tensor &t) {
    if (t.size()) {
      EXPECT_EQ(1, t.ref_count());
    }
  }

  /*
   * Approximately equal numbers.
   */
  template<typename elt_t1, typename elt_t2>
  bool simeq(elt_t1 a, elt_t2 b, double epsilon = 2*EPSILON)
  {
    double x = abs(a - b);
    if (x > epsilon) {
      std::cout << x << std::endl;
      return false;
    }
    return true;
  }

  template<typename elt_t>
  bool simeq(const Tensor<elt_t> &a, const Tensor<elt_t> &b, double epsilon = 2*EPSILON)
  {
    for (typename Tensor<elt_t>::const_iterator ia = a.begin(), ib = b.begin();
         ia != a.end(); ia++, ib++)
      {
      double x = abs(*ia - *ib);
      if (x > epsilon) {
        std::cout << x << std::endl;
        return false;
      }
      return true;
      }
  }

#define EXPECT_CEQ(a, b) EXPECT_TRUE(simeq(a, b))
#define EXPECT_CEQ3(a, b, c) EXPECT_TRUE(simeq(a, b, c))
#define ASSERT_CEQ(a, b) ASSERT_TRUE(simeq(a, b))

  /*
   * Approximately equal tensors.
   */
  template<class Tensor>
  bool approx_eq(const Tensor &A, const Tensor &B, double epsilon = 2*EPSILON)
  {
    if (A.rank() == B.rank()) {
      if (all_equal(A.dimensions(), B.dimensions())) {
        double x = norm0(A - B);
        if (x > epsilon) {
          std::cout << "Deviation: " << x << std::endl;
          return false;
        }
        return true;
      }
    }
    return false;
  }

  template<typename elt_t>
  bool unitaryp(const Tensor<elt_t> &U, double epsilon = EPSILON)
  {
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
  inline void
  test_over_integers(int min, int max, void test(int))
  {
    for (; min <= max; min++) {
      test(min);
    }
  }

  /*
   * Creates a vector of random dimensions.
   */
  inline Indices random_dimensions(int rank, int max_dim) {
    Indices dims(rank);
    for (int i = 0; i < rank; i++) {
      dims.at(i) = rand<int>(0, max_dim+1);
    }
    return dims;
  }

  /*
   * Creates a vector of random dimensions some of which are empty
   */
  inline Indices random_empty_dimensions(int rank, int max_dim, int which = -1) {
    Indices dims = random_dimensions(rank, max_dim);
    if (which < 0 || which >= rank)
      which = rand<int>(rank);
    dims.at(which) = 0;
    return dims;
  }

  /*
   * Loop over dimensions
   */

  class DimensionIterator {
  public:
    DimensionIterator(int rank, int max_dim = 10) :
      dims_(rank), max_(max_dim), more_(true)
    {
      std::fill(dims_.begin(), dims_.end(), 0);
    }

    bool operator++() {
      for (int i = 0; i < dims_.size(); i++) {
        if (++dims_.at(i) < max_) {
          return more_ = true;
        }
        dims_.at(i) = 0;
      }
      return more_ = false;
    }

    const Indices &operator*() const {
      return dims_;
    }

    operator bool() const {
      return more_;
    }

  private:
    Indices dims_;
    int max_;
    bool more_;
  };

  class fixed_rank_iterator {
  public:
    fixed_rank_iterator(int rank, int max_dimension = 10) :
      rank_(rank), max_dimension_(max_dimension),
      indices_(rank), finished_(false)
    {
      assert((rank >= 0) && (max_dimension >= 0));
      reset();
    }
    int rank() const {
      return rank_;
    }
    int max_dimension() const {
      return max_dimension_;
    }
    bool finished() const {
      return finished_;
    }
    bool more() const {
      return !finished_;
    }
    const Indices &dimensions() const {
      return indices_;
    }
    void reset() {
      std::fill(indices_.begin(), indices_.end(), 0);
    }
    bool next() {
      if (!finished_) {
        for (int i = 0; i < rank(); i++) {
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
    
  template<typename elt_t, typename unop>
  void
  test_over_fixed_rank_tensors(unop test, int rank, int max_dimension = 10) {
    for (fixed_rank_iterator it(rank, max_dimension);
         !it.finished(); it.next())
    {
      Tensor<elt_t> data(it.dimensions());
      // Make all elements different to make accurate comparisons
      for (tensor::index i = 0; i < data.size(); i++)
        data.at(i) = i;
      test(data);
    }
  }

  template<typename elt_t, typename binop>
  void
  test_over_fixed_rank_pairs(binop test, int rank, int max_dimension = 6) {
    for (fixed_rank_iterator it1(rank, max_dimension);
         !it1.finished(); it1.next())
    {
      Tensor<elt_t> data1(it1.dimensions());
      // Make all elements different to make accurate comparisons
      for (tensor::index i = 0; i < data1.size(); i++)
        data1.at(i) = i;
      for (fixed_rank_iterator it2(rank, max_dimension);
           !it2.finished(); it2.next())
      {
        Tensor<elt_t> data2(it2.dimensions());
        // Make all elements different to make accurate comparisons
        for (tensor::index i = 0; i < data2.size(); i++)
          data2.at(i) = i;
        test(data1, data2);
      }
    }
  }

  /*
   * Test over all tensor sizes and ranks, randomly.
   */
  template<typename elt_t>
  void
  test_over_all_tensors(void test(Tensor<elt_t> &t), int max_rank = 4,
                        int max_dimension = 10) {
    for (int rank = 0; rank <= max_rank; rank++) {
      char rank_string[] = "rank:      ";
      sprintf(rank_string, "rank: %d", rank);
      SCOPED_TRACE(rank_string);
      test_over_fixed_rank_tensors(test, rank, max_dimension);
    }
  }

  template<typename elt_t>
  void
  test_over_tensors(void test(Tensor<elt_t> &t), int max_rank = 4,
                    int max_dimension = 10, int max_times = 15) {
    for (int rank = 0; rank <= max_rank; rank++) {
      char rank_string[] = "rank:      ";
      sprintf(rank_string, "rank: %d", (int)rank);
      SCOPED_TRACE(rank_string);
      //
      // Test over random dimensions
      //
      for (int times = 0; times < max_times; times++) {
        Tensor<elt_t> data(random_dimensions(rank, max_dimension));
        data.randomize();
        test(data);
      }
      //
      // Forced tests over empty tensors
      //
      for (int times = 0; times < rank; times++) {
        Tensor<elt_t> data(random_empty_dimensions(rank, max_dimension, times));
        test(data);
      }
    }
  }

  class DimensionsProducer {
  public:
    DimensionsProducer(const Indices &d) : base_indices(d), counter(13) {}

    operator bool() const { return counter >= 14; }
    int operator++() { return counter++; }

    Indices operator*() const {
      Tensor<double> P;
      switch (counter) {
        // 1D Tensor<elt_t>
      case 1: P = Tensor<double>(d(0)*d(1)*d(2)*d(3)); break;
        // 2D Tensor<elt_t>
      case 2: P = Tensor<double>(d(0),d(1)*d(2)*d(3)); break;
      case 3: P = Tensor<double>(d(0)*d(1),d(2)*d(3)); break;
      case 4: P = Tensor<double>(d(0)*d(1)*d(2),d(3)); break;
      case 5: P = Tensor<double>(d(3),d(0)*d(1)*d(2)); break;
      case 6: P = Tensor<double>(d(2)*d(3),d(0)*d(1)); break;
        // 3D Tensor<elt_t>
      case 7: P = Tensor<double>(d(0),d(1),d(2)*d(3)); break;
      case 8: P = Tensor<double>(d(0)*d(1),d(2),d(3)); break;
      case 9: P = Tensor<double>(d(3)*d(2),d(0),d(1)); break;
        // 4D Tensor<elt_t>
      case 10: P = Tensor<double>(d(0),d(1),d(2),d(3)); break;
      case 11: P = Tensor<double>(d(3),d(1),d(2),d(0)); break;
      case 12: P = Tensor<double>(d(1),d(0),d(3),d(2)); break;
      case 13: P = Tensor<double>(d(2),d(0),d(3),d(1)); break;
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

  template<typename elt_t> Tensor<elt_t> random_unitary(int n, int iterations = -1);
  template<> Tensor<double> random_unitary(int n, int iterations);
  template<> Tensor<cdouble> random_unitary(int n, int iterations);
  Tensor<double> random_permutation(int n, int iterations);

  static struct Foo { Foo() { tensor::tensor_abort_handler(); }} foo;

} // namespace tensor_test

#endif /* !TENSOR_TEST_LOOPS_H */
