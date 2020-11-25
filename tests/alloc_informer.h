// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_TEST_ALLOC_INFORMER_H
#define TENSOR_TEST_ALLOC_INFORMER_H

class AllocInformer{
 public:
  static int allocations;
  static int deallocations;

  AllocInformer() { allocations++; }
  ~AllocInformer() { deallocations++; }

  static void reset_counters() {
    allocations = deallocations = 0;
  }
};

#endif // !TENSOR_TEST_ALLOC_INFORMER_H
