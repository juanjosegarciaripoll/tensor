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

#define EPSILON 1e-14

#ifdef NDEBUG
#define ONLY_IN_DEBUG(x)
#else
#define ONLY_IN_DEBUG(x) x
#endif

template<typename elt_t>
bool operator==(const tensor::Vector<elt_t> &v1,
                const tensor::Vector<elt_t> &v2)
{
  if (v1.size() != v2.size()) return false;
  return std::equal(v1.begin_const(), v1.end_const(), v2.begin_const());
}

template<typename elt_t, size_t n>
bool operator==(const tensor::StaticVector<elt_t,n> &v1,
                const tensor::Vector<elt_t> &v2)
{
  tensor::Vector<elt_t> v0(v1);
  if (v0.size() != v2.size()) return false;
  return std::equal(v0.begin_const(), v0.end_const(), v2.begin_const());
}

template<typename elt_t, size_t n>
bool operator==(const tensor::Vector<elt_t> &v2,
                const tensor::StaticVector<elt_t,n> &v1)
{
  tensor::Vector<elt_t> v0(v1);
  if (v0.size() != v2.size()) return false;
  return std::equal(v0.begin_const(), v0.end_const(), v2.begin_const());
}

template<size_t n>
bool operator==(const tensor::Indices &v2,
                const tensor::StaticVector<tensor::index,n> &v1)
{
  tensor::Indices v0(v1);
  if (v0.size() != v2.size()) return false;
  return std::equal(v0.begin_const(), v0.end_const(), v2.begin_const());
}

template<size_t n>
bool operator==(const tensor::StaticVector<tensor::index,n> &v1,
                const tensor::Indices &v2)
{
  tensor::Indices v0(v1);
  if (v0.size() != v2.size()) return false;
  return std::equal(v0.begin_const(), v0.end_const(), v2.begin_const());
}

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
   * Approximately equal tensors.
   */
  template<class Tensor>
  bool approx_eq(const Tensor &A, const Tensor &B, double epsilon = 2*EPSILON)
  {
    if (A.rank() == B.rank()) {
      if (A.dimensions() == B.dimensions()) {
        double x = norm0(A - B);
        if (x > epsilon) {
          std::cout << x << std::endl;
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

  template<typename elt_t>
  void
  test_over_fixed_rank_tensors(void test(Tensor<elt_t> &t), int rank,
                               int max_dimension = 10) {
    //
    // Test over random dimensions
    //
    Indices dims(rank);
    std::fill(dims.begin(), dims.end(), 0);
    bool goon = true;
    while (goon) {
      Tensor<elt_t> data(dims);
      // Make all elements different to make accurate comparisons
      for (tensor::index i = 0; i < data.size(); i++)
        data.at(i) = i;
      test(data);
      goon = false;
      for (int i = 0; i < rank; i++) {
        if (++dims.at(i) < max_dimension) {
          goon = true;
          break;
        }
        dims.at(i) = 0;
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
    for (size_t rank = 0; rank <= max_rank; rank++) {
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
    for (size_t rank = 0; rank <= max_rank; rank++) {
      char rank_string[] = "rank:      ";
      sprintf(rank_string, "rank: %d", rank);
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
    int operator++() { counter++; }

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

} // namespace tensor_test

#endif /* !TENSOR_TEST_LOOPS_H */
