#pragma once
/**
sgs.h : Aiming to replace cfg4/CGenstep 
-----------------------------------------
**/


#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SGS_METHOD __host__ __device__ __forceinline__
#else
#    define SGS_METHOD inline 
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
     #include <string>
     #include "spho.h"
#endif


struct sgs
{
    int index ;     // 0-based index of genstep in the event 
    int photons ;   // number of photons in the genstep
    int offset ;    // photon offset in the sequence of gensteps, ie number of photons in event before this genstep
    int gentype  ;  // OpticksGenstep_ enum 
   
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    spho MakePho(unsigned idx, const spho& ancestor); 
    std::string desc() const ; 
#endif
}; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <sstream>
#include <iomanip>
#include "OpticksGenstep.h"

/**
sgs::MakePho
-------------

ancestor.isDefined:false
    the more common case, when generating primary optical 
    photons via the Cerenkov or Scintillation processes 

    HMM: "ancestor" should more correctly be called "reemissionAncestorPhoton"

ancestor.isDefined:true
    when a photon undergoes reemission the ancestor is the parent photon

**/

inline spho sgs::MakePho(unsigned idx, const spho& ancestor)
{
    return ancestor.isDefined() ? ancestor.make_reemit() : spho::MakePho(index, idx, offset + idx, 0) ; 
}

inline std::string sgs::desc() const 
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

