/*
	Copyright (c) 2013 Juan Jose Garcia Ripoll

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
#include <tensor/indices.h>

namespace tensor_test {

// helper function to produce random Booleans
Booleans randomBoolean(int size) {
  Booleans retval(size);

  for (Booleans::iterator it = retval.begin(); it != retval.end(); ++it) {
    *it = (rand<int>() % 2 == 1);
  }

  return retval;
}

TEST(BooleansTest, checkNegation) {
  Booleans input = randomBoolean(10000);
  Booleans output = !input;
  ASSERT_EQ(input.size(), output.size());

  Booleans::iterator it1 = input.begin();
  Booleans::iterator it2 = output.begin();
  for (; it1 != input.end(); ++it1, ++it2) {
    ASSERT_TRUE(*it1 != *it2);
  }
}

TEST(BooleansTest, checkLogicalAnd) {
  Booleans input1 = randomBoolean(10000);
  Booleans input2 = randomBoolean(10000);
  Booleans output = input1 && input2;
  ASSERT_EQ(input1.size(), output.size());

  Booleans::iterator it1 = input1.begin();
  Booleans::iterator it2 = input2.begin();
  for (Booleans::iterator oit = output.begin(); oit != output.end();
       ++it1, ++it2, ++oit) {
    ASSERT_TRUE(*oit == ((*it1) && (*it2)));
  }
}

TEST(BooleansTest, checkLogicalOr) {
  Booleans input1 = randomBoolean(10000);
  Booleans input2 = randomBoolean(10000);
  Booleans output = input1 || input2;
  ASSERT_EQ(input1.size(), output.size());

  Booleans::iterator it1 = input1.begin();
  Booleans::iterator it2 = input2.begin();
  for (Booleans::iterator oit = output.begin(); oit != output.end();
       ++it1, ++it2, ++oit) {
    ASSERT_TRUE(*oit == ((*it1) || (*it2)));
  }
}

#ifdef TENSOR_DEBUG
// death by assert
TEST(BooleansTest, deathOnWrongSizes) {
  ASSERT_THROW_DEBUG(randomBoolean(10) && randomBoolean(11),
                     ::tensor::invalid_assertion);
  ASSERT_THROW_DEBUG(randomBoolean(10) || randomBoolean(11),
                     ::tensor::invalid_assertion);
}
#endif

TEST(BooleansTest, checkFindingOfIndices) {
  Booleans input = randomBoolean(10000);
  Indices output = which(input);

  Indices::iterator it = output.begin();
  for (int i = 0; i < input.size(); ++i) {
    if (input[i]) {
      ASSERT_EQ(i, *it);
      ++it;
    }
  }
}

TEST(BooleansTest, checkAllOf) {
  Booleans input = Booleans(10000);

  std::fill(input.begin(), input.end(), true);
  ASSERT_TRUE(all_of(input));

  input.at(5) = false;
  ASSERT_FALSE(all_of(input));
}

TEST(BooleansTest, checkAnyAndNoneOf) {
  Booleans input = Booleans(10000);

  std::fill(input.begin(), input.end(), false);
  ASSERT_FALSE(any_of(input));
  ASSERT_TRUE(none_of(input));

  input.at(0) = true;
  ASSERT_TRUE(any_of(input));
  ASSERT_FALSE(none_of(input));
}

}  // namespace tensor_test
