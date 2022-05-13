#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SGS_METHOD __host__ __device__ __forceinline__
#else
#    define SGS_METHOD inline 
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
     #include <string>
#endif

struct sgs
{
    unsigned index ;     // 0-based index of genstep in the event 
    unsigned photons ;   // number of photons in the genstep
    unsigned offset ;    // photon offset in the sequence of gensteps, ie number of photons in event before this genstep
    unsigned gentype  ;  // OpticksGenstep_ enum 
   
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string desc() const ; 
#endif
}; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <sstream>
#include <iomanip>
#include "OpticksGenstep.h"

std::string sgs::desc() const 
{
    std::stringstream ss ; 
    ss << "sgs:"
       << " idx" << std::setw(4) << index 
       << " pho" << std::setw(6) << photons 
       << " off " << std::setw(6) << offset 
       << " typ " << OpticksGenstep_::Name(gentype)
       ;   
    std::string s = ss.str(); 
    return s ; 
}
#endif

