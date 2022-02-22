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

#include <cstdlib>
#if defined(_MSC_VER) || defined(__MINGW32__)
#include <windows.h>
#include <Wincrypt.h>
#include <time.h>
#else
#include <cstdio>
#endif
#include <tensor/rand.h>
#include "mt.h"

namespace tensor {

static bool initialize_mt() {
  rand_reseed();
  return true;
}

#ifndef SEED_SIZE
#define SEED_SIZE 1
#endif

void set_seed(unsigned long seed) { init_genrand(seed); }

void rand_reseed() {
#if defined(_MSC_VER) || defined(__MINGW32__)
#if 1
  // The following code has a problem: it requires additional libraries
  HCRYPTPROV hCryptProv;
  union {
    BYTE data[SEED_SIZE * sizeof(rand_uint)];
    rand_uint seed[SEED_SIZE];
  } r;
  int ok = 0;
  if (CryptAcquireContext(&hCryptProv, nullptr, nullptr, PROV_RSA_FULL, 0)) {
    ok = CryptGenRandom(hCryptProv, sizeof(r.seed), r.data);
    CryptReleaseContext(hCryptProv, 0);
  }
  if (!ok) {
    init_genrand(static_cast<uint32_t>(clock()) ^
                 static_cast<uint32_t>(time(0)));
  }
#else
  // Sleep one second at least to get a different value on each call.
  static int firsttime = 1;
  if (!firsttime) {
    Sleep(1000);
  }
  firsttime = 0;
  init_genrand((uint32_t)clock() ^ (uint32_t)time(0));
#endif
#else
  char *rand_seed = getenv("RANDSEED");
  if (rand_seed) {
    int seed = atoi(rand_seed);
    std::cout << "RANDSEED=" << seed << std::endl;
    init_genrand(static_cast<uint64_t>(seed));
    return;
  } else {
    FILE *fp = fopen("/dev/urandom", "r");
    rand_uint seed[SEED_SIZE];
    if (fp && (SEED_SIZE > 0)) {
      fread(&seed, sizeof(rand_uint), SEED_SIZE, fp);
      fclose(fp);
      init_by_array(seed, SEED_SIZE);
    } else {
      auto aseed = static_cast<uint64_t>(time(0));
      init_genrand(static_cast<uint64_t>(aseed));
    }
  }
#endif
  for (size_t i = 0; i < 624 * 10; i++) {
    // Warm up
    rand<long>();
  }
}

static bool mt_initialized = initialize_mt();

#ifdef TENSOR_64BITS
template <>
int rand<int>() {
  return static_cast<int>(genrand_int63());
}
template <>
unsigned int rand<unsigned int>() {
  return static_cast<unsigned int>(genrand_int64());
}
template <>
long rand<long>() {
  return static_cast<long>(genrand_int63());
}
template <>
unsigned long rand<unsigned long>() {
  return static_cast<unsigned long>(genrand_int64());
}
#else
template <>
int rand<int>() {
  return genrand_int31();
}
template <>
unsigned int rand<unsigned int>() {
  return genrand_int32();
}
template <>
long rand<long>() {
  return genrand_int31();
}
template <>
unsigned long rand<unsigned long>() {
  return genrand_int32();
}
#endif

template <>
float rand<float>() {
  return static_cast<float>(genrand_res53());
}

template <>
double rand<double>() {
  return genrand_res53();
}

template <>
cdouble rand<cdouble>() {
  return to_complex(genrand_res53(), genrand_res53());
}

}  // namespace tensor
