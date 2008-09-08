// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_TEST_LOOPS_H
#define TENSOR_TEST_LOOPS_H

#include <algorithm>
#include <gtest/gtest.h>
#include <tensor/rand.h>
#include <tensor/tensor.h>
#include <tensor/io.h>

#define EPSILON 1e-14

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
 * Test over all tensor sizes and ranks, randomly.
 */
template<typename elt_t>
void
test_over_tensors(void test(Tensor<elt_t> &t), int max_rank = 4, int max_dimension = 10) {
  for (size_t rank = 1; rank <= max_rank; rank++) {
    char rank_string[] = "rank:      ";
    sprintf(rank_string, "rank: %d", rank);
    SCOPED_TRACE(rank_string);
    for (size_t times = 0; times < 15; times++) {
      Tensor<elt_t> data(random_dimensions(max_rank, max_dimension));
      data.randomize();
      test(data);
    }
  }
}

} // namespace tensor_test

#endif /* !TENSOR_TEST_LOOPS_H */
