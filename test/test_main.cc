// -*- mode: c++; fill-column: 80; c-basic-offset: 4; -*-
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
}
