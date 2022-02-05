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

#include <cstdio>
#include <iostream>
#include <tensor/sdf.h>
#include <gtest/gtest.h>

using namespace sdf;

TEST(SDF, RTensor) {
  RTensor a = RTensor::random(1);
  RTensor b = RTensor::random(13);
  RTensor c = RTensor::random(4, 15);
  RTensor d = RTensor::random(3, 7, 5);
  RTensor x = {1e-312};
  if (file_exists("sdf_data_rtensor.dat")) {
    std::remove("sdf_data_rtensor.dat");
  }
  {
    OutDataFile f("sdf_data_rtensor.dat");
    f.dump(a, "a");
    f.dump(a[0], "a0");
    f.dump(b, "b");
    f.dump(c, "c");
    f.dump(d, "d");
    f.dump(x, "x");
  }
  {
    InDataFile f("sdf_data_rtensor.dat");
    RTensor aux;
    f.load(&aux, "a");
    EXPECT_TRUE(all_equal(a, aux));
    double a0;
    f.load(&a0, "a0");
    EXPECT_EQ(a0, a[0]);
    f.load(&aux, "b");
    EXPECT_TRUE(all_equal(b, aux));
    f.load(&aux, "c");
    EXPECT_TRUE(all_equal(c, aux));
    f.load(&aux, "d");
    EXPECT_TRUE(all_equal(d, aux));
    f.load(&aux, "x");
    EXPECT_TRUE(all_equal(x, aux));
  }
  std::remove("sdf_data_rtensor.dat");
}

TEST(SDF, CTensor) {
  CTensor a = CTensor::zeros(1);
  CTensor b = CTensor::random(13);
  CTensor c = CTensor::random(4, 15);
  CTensor d = CTensor::random(3, 7, 5);
  if (file_exists("sdf_data_ctensor.dat")) {
    std::remove("sdf_data_ctensor.dat");
  }
  {
    OutDataFile f("sdf_data_ctensor.dat");
    f.dump(a, "a");
    f.dump(a[0], "a0");
    f.dump(b, "b");
    f.dump(c, "c");
    f.dump(d, "d");
  }
  {
    InDataFile f("sdf_data_ctensor.dat");
    CTensor aux;
    f.load(&aux, "a");
    EXPECT_TRUE(all_equal(a, aux));
    cdouble a0;
    f.load(&a0, "a0");
    EXPECT_EQ(a0, a[0]);
    f.load(&aux, "b");
    EXPECT_TRUE(all_equal(b, aux));
    f.load(&aux, "c");
    EXPECT_TRUE(all_equal(c, aux));
    f.load(&aux, "d");
    EXPECT_TRUE(all_equal(d, aux));
  }
  std::remove("sdf_data_ctensor.dat");
}
