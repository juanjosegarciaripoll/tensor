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

#include <tensor/jobs.h>
#include <gtest/gtest.h>

static bool zerop(int n) { return n == 0; }

TEST(Jobs, ErrorCheck) {
  {
    int argc = 1;
    const char *argv[] = {"--variable"};
    ASSERT_DEATH(jobs::Job(argc, argv), ".*");
  }
  {
    int argc = 1;
    const char *argv[] = {"--this-job"};
    ASSERT_DEATH(jobs::Job(argc, argv), ".*");
  }
  {
    int argc = 1;
    const char *argv[] = {"--help"};
    ASSERT_EXIT(jobs::Job(argc, argv), zerop, ".*");
  }
}

TEST(Jobs, FixedVariable) {
  int argc = 2;
  const char *argv[] = {"--variable", "var,0.13"};
  jobs::Job job(argc, argv);
  EXPECT_DOUBLE_EQ(job.get_value("var"), 0.13);
}

TEST(Jobs, DefaultRange) {
  {
    int argc = 2;
    const char *argv[] = {"--variable", "var,0,9"};
    EXPECT_DOUBLE_EQ(jobs::Job(argc, argv).get_value("var"), 0);
  }
  {
    int argc = 4;
    const char *argv[] = {"--this-job", "3", "--variable", "var,0,9"};
    EXPECT_DOUBLE_EQ(jobs::Job(argc, argv).get_value("var"), 3.0);
  }
  {
    int argc = 4;
    const char *argv[] = {"--this-job", "7", "--variable", "var,0,9"};
    EXPECT_DOUBLE_EQ(jobs::Job(argc, argv).get_value("var"), 7.0);
  }
}

TEST(Jobs, ProvidedRange) {
  {
    int argc = 2;
    const char *argv[] = {"--variable", "var,0,19,20"};
    EXPECT_DOUBLE_EQ(jobs::Job(argc, argv).get_value("var"), 0);
  }
  {
    int argc = 4;
    const char *argv[] = {"--this-job", "3", "--variable", "var,0,19,20"};
    EXPECT_DOUBLE_EQ(jobs::Job(argc, argv).get_value("var"), 3.0);
  }
  {
    int argc = 4;
    const char *argv[] = {"--this-job", "7", "--variable", "var,0,19,20"};
    EXPECT_DOUBLE_EQ(jobs::Job(argc, argv).get_value("var"), 7.0);
  }
}
