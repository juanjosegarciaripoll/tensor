// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//
#ifndef TENSOR_RAND_MT_H
#define TENSOR_RAND_MT_H

#include "config.h"
#include <inttypes.h>

namespace tensor {

#if !defined(TENSOR_64BITS)

/*
 * 32 bits
 */

typedef uint32_t rand_uint;

extern void init_genrand(uint32_t s);
extern void init_by_array(uint32_t init_key[], int key_length);

/* generates a random number on [0,0xffffffff]-interval */
extern uint32_t genrand_int32(void);

/* generates a random number on [0,0x7fffffff]-interval */
extern int32_t genrand_int31(void);

#else

/*
 * 64 bits
 */

typedef uint64_t rand_uint;

extern void init_genrand(uint64_t s);
extern void init_by_array(uint64_t init_key[], int key_length);

/* generates a random number on [0,0xffffffff]-interval */
extern uint64_t genrand_int64(void);

/* generates a random number on [0,0x7fffffff]-interval */
extern int64_t genrand_int63(void);

#endif

/* generates a random number on [0,1]-real-interval */
extern double genrand_real1(void);

/* generates a random number on [0,1)-real-interval */
extern double genrand_real2(void);

/* generates a random number on (0,1)-real-interval */
extern double genrand_real3(void);

/* generates a random number on [0,1) with 53-bit resolution*/
extern double genrand_res53(void);

} // namespace tensor

#endif // !TENSOR_RAND_MT_H

