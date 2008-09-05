// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#include <gtest/gtest.h>
#include <tensor/refcount.h>

int
main(int narg, char *argv[])
{
  ::testing::InitGoogleTest(&narg, argv);

  RUN_ALL_TESTS();

  return 0;
}
