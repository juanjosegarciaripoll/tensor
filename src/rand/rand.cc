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
#include <cstdlib>
#include <tensor/rand.h>
#include "mt.h"

namespace tensor {

static bool
initialize_mt()
{
  rand_reseed();
  return true;
}

void rand_reseed() {
  char *rand_seed = getenv("RANDSEED");
  if (rand_seed) {
    int seed = atoi(rand_seed);
    std::cout << "RANDSEED=" << seed << std::endl;
    init_genrand(seed);
    return;
  } else {
    FILE *fp = fopen("/dev/urandom", "r");
#ifndef SEED_SIZE
#define SEED_SIZE 1
#endif
    rand_uint seed[SEED_SIZE];
    if (fp && (SEED_SIZE > 0)) {
      fread(&seed, sizeof(rand_uint), SEED_SIZE, fp);
      fclose(fp);
      init_by_array(seed, SEED_SIZE);
    } else {
      int seed = time(0);
      init_genrand(seed);
    }
  }
  for (size_t i = 0; i < 624*10; i++) {
    // Warm up
    rand<long>();
  }
}

static bool mt_initialized = initialize_mt();

#ifdef TENSOR_64BITS
template<> int rand<int>() { return genrand_int63(); }
template<> unsigned int rand<unsigned int>() { return genrand_int64(); }
template<> long rand<long>() { return genrand_int63(); }
template<> unsigned long rand<unsigned long>() { return genrand_int64(); }
#else
template<> int rand<int>() { return genrand_int31(); }
template<> unsigned int rand<unsigned int>() { return genrand_int32(); }
template<> long rand<long>() { return genrand_int31(); }
template<> unsigned long rand<unsigned long>() { return genrand_int32(); }
#endif

template<> float rand<float>() {
  return genrand_res53();
}

template<> double rand<double>() {
  return genrand_res53();
}

template<> cdouble rand<cdouble>() {
  return to_complex(genrand_res53(), genrand_res53());
}

} // namespace rand
