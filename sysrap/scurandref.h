#pragma once
#include "curand_kernel.h"

/**
chunk_idx
   index of the chunk
chunk_offset
   number of state slots prior to this chunk 

num
   number of state slots in the *chunk_idx* chunk

seed
   input to curand_init, default 0 
offset 
   input to curand_init, default 0 


**/


struct scurandref
{
    unsigned long long chunk_idx ; 
    unsigned long long chunk_offset ;

    unsigned long long num ; 
    unsigned long long seed ;
    unsigned long long offset ;
    curandState*  states  ; 
};

