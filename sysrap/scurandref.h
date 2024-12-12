#pragma once

/**
scurandref : NOT GENERAL : SPECIFIC TO curandStateXORWOW
===========================================================

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

#include "curand_kernel.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#endif


template<typename T>
struct scurandref
{
    unsigned long long chunk_idx ; 
    unsigned long long chunk_offset ;

    unsigned long long num ; 
    unsigned long long seed ;
    unsigned long long offset ;
    T*                 states  ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string desc() const ; 
#endif

};


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template<typename T>
inline std::string scurandref<T>::desc() const 
{
    std::stringstream ss ; 
    ss << "scurandref::desc"
       << " chunk_idx " << std::setw(4) << chunk_idx 
       << " chunk_offset/M " << std::setw(4) << chunk_offset/1000000
       << " num/M " << std::setw(4) << num/1000000
       << " seed " << seed
       << " offset " << offset
       << " states " << states 
       << "\n"
       ;
    std::string str = ss.str() ; 
    return str ; 
}
#endif



