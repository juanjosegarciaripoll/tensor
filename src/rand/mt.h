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
#ifndef TENSOR_RAND_MT_H
#define TENSOR_RAND_MT_H

#include "tensor/config.h"
#if defined(_MSC_VER) && (_MSC_VER < 1800)
#include <stdint.h>
#else
#include <inttypes.h>
#endif

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

