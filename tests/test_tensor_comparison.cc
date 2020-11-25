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

#include "loops.h"
#include <gtest/gtest.h>
#include <tensor/tensor.h>

namespace tensor_test {

  //////////////////////////////////////////////////////////////////////
  // RTENSOR - RTENSOR SPECIALIZATIONS
  //

  TEST(TensorCompare, RTensorRTensorEqual) {

    EXPECT_EQ(Booleans(), RTensor() == RTensor());

    EXPECT_EQ(bgen << true, (rgen << 1.0) == (rgen << 1.0));
    EXPECT_EQ(bgen << false, (rgen << 1.0) == (rgen << 0.0));

    EXPECT_EQ(bgen << true << true,
	      (rgen << 1.0 << 2.0) == (rgen << 1.0 << 2.0));
    EXPECT_EQ(bgen << true << false,
	      (rgen << 1.0 << 2.0) == (rgen << 1.0 << 1.0));
    EXPECT_EQ(bgen << false << true,
	      (rgen << 1.0 << 2.0) == (rgen << 2.0 << 2.0));
    EXPECT_EQ(bgen << false << false,
	      (rgen << 1.0 << 2.0) == (rgen << 2.0 << 1.0));

    EXPECT_EQ(bgen << true << true << true << true,
	      RTensor::ones(2,2) == RTensor::ones(2,2));
  }

  TEST(TensorCompare, RTensorRTensorNotEqual) {

    EXPECT_EQ(Booleans(), RTensor() != RTensor());

    EXPECT_EQ(bgen << false, (rgen << 1.0) != (rgen << 1.0));
    EXPECT_EQ(bgen << true, (rgen << 1.0) != (rgen << 0.0));

    EXPECT_EQ(bgen << false << false,
	      (rgen << 1.0 << 2.0) != (rgen << 1.0 << 2.0));
    EXPECT_EQ(bgen << false << true,
	      (rgen << 1.0 << 2.0) != (rgen << 1.0 << 1.0));
    EXPECT_EQ(bgen << true << false,
	      (rgen << 1.0 << 2.0) != (rgen << 2.0 << 2.0));
    EXPECT_EQ(bgen << true << true,
	      (rgen << 1.0 << 2.0) != (rgen << 2.0 << 1.0));

    EXPECT_EQ(bgen << false << false << false << false,
	      RTensor::ones(2,2) != RTensor::ones(2,2));
  }

  TEST(TensorCompare, RTensorRTensorLess) {

    EXPECT_EQ(Booleans(), RTensor() < RTensor());

    EXPECT_EQ(bgen << false, (rgen << 1.0) < (rgen << 1.0));
    EXPECT_EQ(bgen << false, (rgen << 1.0) < (rgen << 0.0));
    EXPECT_EQ(bgen << true, (rgen << 1.0) < (rgen << 2.0));

    EXPECT_EQ(bgen << false << false,
	      (rgen << 1.0 << 2.0) < (rgen << 1.0 << 2.0));
    EXPECT_EQ(bgen << false << false,
	      (rgen << 1.0 << 2.0) < (rgen << 1.0 << 1.0));
    EXPECT_EQ(bgen << false << false,
	      (rgen << 1.0 << 2.0) < (rgen << 0.0 << 2.0));
    EXPECT_EQ(bgen << false << false,
	      (rgen << 1.0 << 2.0) < (rgen << 0.0 << 0.0));
    EXPECT_EQ(bgen << true << false,
	      (rgen << 1.0 << 2.0) < (rgen << 2.0 << 1.0));
    EXPECT_EQ(bgen << false << true,
	      (rgen << 1.0 << 2.0) < (rgen << 1.0 << 3.0));
    EXPECT_EQ(bgen << true << true,
	      (rgen << 1.0 << 2.0) < (rgen << 2.0 << 3.0));

    EXPECT_EQ(bgen << true << true << true << true,
	      RTensor::ones(2,2) < 2.0 * RTensor::ones(2,2));
    EXPECT_EQ(bgen << false << false << false << false,
	      RTensor::ones(2,2) < RTensor::ones(2,2));
  }

  TEST(TensorCompare, RTensorRTensorGreater) {

    EXPECT_EQ(Booleans(), RTensor() > RTensor());

    EXPECT_EQ(bgen << false, (rgen << 1.0) > (rgen << 1.0));
    EXPECT_EQ(bgen << true, (rgen << 1.0) > (rgen << 0.0));
    EXPECT_EQ(bgen << false, (rgen << 1.0) > (rgen << 2.0));

    EXPECT_EQ(bgen << false << false,
	      (rgen << 1.0 << 2.0) > (rgen << 1.0 << 2.0));
    EXPECT_EQ(bgen << false << true,
	      (rgen << 1.0 << 2.0) > (rgen << 1.0 << 1.0));
    EXPECT_EQ(bgen << true << false,
	      (rgen << 1.0 << 2.0) > (rgen << 0.0 << 2.0));
    EXPECT_EQ(bgen << true << true,
	      (rgen << 1.0 << 2.0) > (rgen << 0.0 << 0.0));
    EXPECT_EQ(bgen << false << true,
	      (rgen << 1.0 << 2.0) > (rgen << 2.0 << 1.0));
    EXPECT_EQ(bgen << false << false,
	      (rgen << 1.0 << 2.0) > (rgen << 1.0 << 3.0));
    EXPECT_EQ(bgen << false << false,
	      (rgen << 1.0 << 2.0) > (rgen << 2.0 << 3.0));

    EXPECT_EQ(bgen << true << true << true << true,
	      2.0 * RTensor::ones(2,2) > RTensor::ones(2,2));
    EXPECT_EQ(bgen << false << false << false << false,
	      RTensor::ones(2,2) > RTensor::ones(2,2));
  }

  TEST(TensorCompare, RTensorRTensorLessEqual) {

    EXPECT_EQ(Booleans(), RTensor() <= RTensor());

    EXPECT_EQ(bgen << true, (rgen << 1.0) <= (rgen << 1.0));
    EXPECT_EQ(bgen << false, (rgen << 1.0) <= (rgen << 0.0));
    EXPECT_EQ(bgen << true, (rgen << 1.0) <= (rgen << 2.0));

    EXPECT_EQ(bgen << true << true,
	      (rgen << 1.0 << 2.0) <= (rgen << 1.0 << 2.0));
    EXPECT_EQ(bgen << true << false,
	      (rgen << 1.0 << 2.0) <= (rgen << 1.0 << 1.0));
    EXPECT_EQ(bgen << false << true,
	      (rgen << 1.0 << 2.0) <= (rgen << 0.0 << 2.0));
    EXPECT_EQ(bgen << false << false,
	      (rgen << 1.0 << 2.0) <= (rgen << 0.0 << 0.0));
    EXPECT_EQ(bgen << true << false,
	      (rgen << 1.0 << 2.0) <= (rgen << 2.0 << 1.0));
    EXPECT_EQ(bgen << true << true,
	      (rgen << 1.0 << 2.0) <= (rgen << 1.0 << 3.0));
    EXPECT_EQ(bgen << true << true,
	      (rgen << 1.0 << 2.0) <= (rgen << 2.0 << 3.0));

    EXPECT_EQ(bgen << true << true << true << true,
	      RTensor::ones(2,2) <= 2.0 * RTensor::ones(2,2));
    EXPECT_EQ(bgen << true << true << true << true,
	      RTensor::ones(2,2) <= RTensor::ones(2,2));
  }

  TEST(TensorCompare, RTensorRTensorGreaterEqual) {

    EXPECT_EQ(Booleans(), RTensor() >= RTensor());

    EXPECT_EQ(bgen << true, (rgen << 1.0) >= (rgen << 1.0));
    EXPECT_EQ(bgen << true, (rgen << 1.0) >= (rgen << 0.0));
    EXPECT_EQ(bgen << false, (rgen << 1.0) >= (rgen << 2.0));

    EXPECT_EQ(bgen << true << true,
	      (rgen << 1.0 << 2.0) >= (rgen << 1.0 << 2.0));
    EXPECT_EQ(bgen << true << true,
	      (rgen << 1.0 << 2.0) >= (rgen << 1.0 << 1.0));
    EXPECT_EQ(bgen << true << true,
	      (rgen << 1.0 << 2.0) >= (rgen << 0.0 << 2.0));
    EXPECT_EQ(bgen << true << true,
	      (rgen << 1.0 << 2.0) >= (rgen << 0.0 << 0.0));
    EXPECT_EQ(bgen << false << true,
	      (rgen << 1.0 << 2.0) >= (rgen << 2.0 << 1.0));
    EXPECT_EQ(bgen << true << false,
	      (rgen << 1.0 << 2.0) >= (rgen << 1.0 << 3.0));
    EXPECT_EQ(bgen << false << false,
	      (rgen << 1.0 << 2.0) >= (rgen << 2.0 << 3.0));

    EXPECT_EQ(bgen << true << true << true << true,
	      2.0 * RTensor::ones(2,2) >= RTensor::ones(2,2));
    EXPECT_EQ(bgen << true << true << true << true,
	      RTensor::ones(2,2) >= RTensor::ones(2,2));
  }


  //////////////////////////////////////////////////////////////////////
  // RTENSOR - DOUBLE SPECIALIZATIONS
  //

  TEST(TensorCompare, RTensorDoubleEqual) {

    EXPECT_EQ(Booleans(), RTensor() == 0.0);

    EXPECT_EQ(bgen << true, (rgen << 1.0) == 1.0);
    EXPECT_EQ(bgen << false, (rgen << 1.0) == 0.0);

    EXPECT_EQ(bgen << true << false, (rgen << 1.0 << 2.0) == 1.0);
    EXPECT_EQ(bgen << false << false, (rgen << 1.0 << 2.0) == 0.0);
    EXPECT_EQ(bgen << false << true, (rgen << 1.0 << 2.0) == 2.0);

    EXPECT_EQ(bgen << true << true << true << true,
	      RTensor::ones(2,2) == 1.0);
  }

  TEST(TensorCompare, RTensorDoubleNotEqual) {

    EXPECT_EQ(Booleans(), RTensor() != 0.0);

    EXPECT_EQ(bgen << false, (rgen << 1.0) != 1.0);
    EXPECT_EQ(bgen << true, (rgen << 1.0) != 0.0);

    EXPECT_EQ(bgen << false << true, (rgen << 1.0 << 2.0) != 1.0);
    EXPECT_EQ(bgen << true << true, (rgen << 1.0 << 2.0) != 0.0);
    EXPECT_EQ(bgen << true << false, (rgen << 1.0 << 2.0) != 2.0);

    EXPECT_EQ(bgen << false << false << false << false,
	      RTensor::ones(2,2) != 1.0);
  }

  TEST(TensorCompare, RTensorDoubleLess) {

    EXPECT_EQ(Booleans(), RTensor() < 1.0);

    EXPECT_EQ(bgen << false, (rgen << 1.0) < 0.0);
    EXPECT_EQ(bgen << false, (rgen << 1.0) < 1.0);
    EXPECT_EQ(bgen << true, (rgen << 1.0) < 2.0);

    EXPECT_EQ(bgen << false << false, (rgen << 1.0 << 2.0) < 0.0);
    EXPECT_EQ(bgen << false << false, (rgen << 1.0 << 2.0) < 1.0);
    EXPECT_EQ(bgen << true << false, (rgen << 1.0 << 2.0) < 2.0);
    EXPECT_EQ(bgen << true << true, (rgen << 1.0 << 2.0) < 3.0);

    EXPECT_EQ(bgen << false << false << false << false,
	      RTensor::ones(2,2) < 0.0);
    EXPECT_EQ(bgen << false << false << false << false,
	      RTensor::ones(2,2) < 1.0);
    EXPECT_EQ(bgen << true << true << true << true,
	      RTensor::ones(2,2) < 2.0);
  }

  TEST(TensorCompare, RTensorDoubleGreater) {

    EXPECT_EQ(Booleans(), RTensor() > 1.0);

    EXPECT_EQ(bgen << true, (rgen << 1.0) > 0.0);
    EXPECT_EQ(bgen << false, (rgen << 1.0) > 1.0);
    EXPECT_EQ(bgen << false, (rgen << 1.0) > 2.0);

    EXPECT_EQ(bgen << true << true, (rgen << 1.0 << 2.0) > 0.0);
    EXPECT_EQ(bgen << false << true, (rgen << 1.0 << 2.0) > 1.0);
    EXPECT_EQ(bgen << false << false, (rgen << 1.0 << 2.0) > 2.0);
    EXPECT_EQ(bgen << false << false, (rgen << 1.0 << 2.0) > 3.0);

    EXPECT_EQ(bgen << true << true << true << true,
	      RTensor::ones(2,2) > 0.0);
    EXPECT_EQ(bgen << false << false << false << false,
	      RTensor::ones(2,2) > 1.0);
    EXPECT_EQ(bgen << false << false << false << false,
	      RTensor::ones(2,2) > 2.0);
  }

  TEST(TensorCompare, RTensorDoubleLessEqual) {

    EXPECT_EQ(Booleans(), RTensor() <= 1.0);

    EXPECT_EQ(bgen << false, (rgen << 1.0) <= 0.0);
    EXPECT_EQ(bgen << true, (rgen << 1.0) <= 1.0);
    EXPECT_EQ(bgen << true, (rgen << 1.0) <= 2.0);

    EXPECT_EQ(bgen << false << false, (rgen << 1.0 << 2.0) <= 0.0);
    EXPECT_EQ(bgen << true << false, (rgen << 1.0 << 2.0) <= 1.0);
    EXPECT_EQ(bgen << true << true, (rgen << 1.0 << 2.0) <= 2.0);
    EXPECT_EQ(bgen << true << true, (rgen << 1.0 << 2.0) <= 3.0);

    EXPECT_EQ(bgen << false << false << false << false,
	      RTensor::ones(2,2) <= 0.0);
    EXPECT_EQ(bgen << true << true << true << true,
	      RTensor::ones(2,2) <= 1.0);
    EXPECT_EQ(bgen << true << true << true << true,
	      RTensor::ones(2,2) <= 2.0);
  }

  TEST(TensorCompare, RTensorDoubleGreaterEqual) {

    EXPECT_EQ(Booleans(), RTensor() >= 1.0);

    EXPECT_EQ(bgen << true, (rgen << 1.0) >= 0.0);
    EXPECT_EQ(bgen << true, (rgen << 1.0) >= 1.0);
    EXPECT_EQ(bgen << false, (rgen << 1.0) >= 2.0);

    EXPECT_EQ(bgen << true << true, (rgen << 1.0 << 2.0) >= 0.0);
    EXPECT_EQ(bgen << true << true, (rgen << 1.0 << 2.0) >= 1.0);
    EXPECT_EQ(bgen << false << true, (rgen << 1.0 << 2.0) >= 2.0);
    EXPECT_EQ(bgen << false << false, (rgen << 1.0 << 2.0) >= 3.0);

    EXPECT_EQ(bgen << true << true << true << true,
	      RTensor::ones(2,2) >= 0.0);
    EXPECT_EQ(bgen << true << true << true << true,
	      RTensor::ones(2,2) >= 1.0);
    EXPECT_EQ(bgen << false << false << false << false,
	      RTensor::ones(2,2) >= 2.0);
  }

} // namespace tensor_test
