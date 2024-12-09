#pragma once
/**
curandStatePhilox4_32_10_OpticksLite.h
========================================

Experiment with counter based RNG, see notes::

    ~/o/notes/curand-impl-review-and-compare-to-curand-done-right.rst

This specializes the curandStatePhilox4_32_10 impl to use minimal counter only state::

   /usr/local/cuda/include/curand_kernel.h
   /usr/local/cuda/include/curand_philox4x32_x.h

As inspired by::

    https://github.com/kshitijl/curand-done-right

+---------------------------------------+----------------+--------------------------------------------------+
|                                       |  sizeof bytes  |   notes                                          |
+=======================================+================+==================================================+
| curandStateXORWOW                     |    48          |  curand default, expensive init => complications |
+---------------------------------------+----------------+--------------------------------------------------+
| curandStatePhilox4_32_10              |    64          |  cheap init (TODO: check in practice)            |
+---------------------------------------+----------------+--------------------------------------------------+
| curandStatePhilox4_32_10_OpticksLite  |    32          |  slim state to uint4 + uint2, gets padded to 32  |
+---------------------------------------+----------------+--------------------------------------------------+

See LICENSE.txt for usage conditions.

Related tests::

   ~/o/sysrap/tests/curand_uniform_test.sh
   ~/o/sysrap/tests/curanddr_uniform_test.sh


**/

#include "curand_kernel.h"


struct curandStatePhilox4_32_10_OpticksLite
{
    uint4 ctr ; 
    uint2 key ;   
    // looks like 6*4=24 bytes, but gets padded to 32 bytes
};

QUALIFIERS void Philox_State_Incr(curandStatePhilox4_32_10_OpticksLite* s)
{
   if(++s->ctr.x) return;
   if(++s->ctr.y) return;
   if(++s->ctr.z) return;
   ++s->ctr.w;
}

QUALIFIERS void Philox_State_Incr(curandStatePhilox4_32_10_OpticksLite* s, unsigned long long n)
{
   unsigned int nlo = (unsigned int)(n);
   unsigned int nhi = (unsigned int)(n>>32);

   s->ctr.x += nlo;
   if( s->ctr.x < nlo )
      nhi++;

   s->ctr.y += nhi;
   if(nhi <= s->ctr.y)
      return;
   if(++s->ctr.z) return;
   ++s->ctr.w;
}

QUALIFIERS void Philox_State_Incr_hi(curandStatePhilox4_32_10_OpticksLite* s, unsigned long long n)
{
   unsigned int nlo = (unsigned int)(n);
   unsigned int nhi = (unsigned int)(n>>32);

   s->ctr.z += nlo;
   if( s->ctr.z < nlo )
      nhi++;

   s->ctr.w += nhi;
}

QUALIFIERS void skipahead_sequence(unsigned long long n, curandStatePhilox4_32_10_OpticksLite* s)
{
    Philox_State_Incr_hi(s, n);
}

QUALIFIERS void skipahead(unsigned long long n, curandStatePhilox4_32_10_OpticksLite* s)
{
    Philox_State_Incr(s, n);
}


QUALIFIERS void curand_init( unsigned long long seed,
                             unsigned long long subsequence,
                             unsigned long long offset,
                             curandStatePhilox4_32_10_OpticksLite* s )
{
    s->ctr = make_uint4(0, 0, 0, 0);
    s->key.x = (unsigned int)seed;
    s->key.y = (unsigned int)(seed>>32);

    skipahead_sequence(subsequence, s);
    skipahead(offset, s);
}

QUALIFIERS float4 curand_uniform4( curandStatePhilox4_32_10_OpticksLite* s )
{
   uint4 result = curand_Philox4x32_10(s->ctr, s->key);  
   Philox_State_Incr(s); 
   return _curand_uniform4(result) ; 
}
/**
curand_uniform
-----------------

This wastefully only uses 1 of the 4 uint generated, 
prefer instead to use curand_uniform4

**/
QUALIFIERS float curand_uniform( curandStatePhilox4_32_10_OpticksLite* s )
{
   uint4 result = curand_Philox4x32_10(s->ctr, s->key);  
   Philox_State_Incr(s); 
   return _curand_uniform(result.x) ;    
}


/**
curand_uniform4(curandStateXORWOW* state)
-------------------------------------------

API missed from XORWOW, added to allow templated 
tests comparing between curandState types::

    curandStateXORWOW
    curandStatePhilox4_32_10
    curandStatePhilox4_32_10_OpticksLite

**/

QUALIFIERS float4 curand_uniform4(curandStateXORWOW* state)
{ 
    float4 result ; 
    result.x = curand_uniform(state); 
    result.y = curand_uniform(state); 
    result.z = curand_uniform(state); 
    result.w = curand_uniform(state); 
    return result ; 
}

