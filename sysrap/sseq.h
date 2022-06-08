#pragma once
/**
sseq.h : photon level step-by-step history and material recording seqhis/seqmat using 64 bit uint
==================================================================================================

For persisting srec arrays use::

   NP* seq = NP<unsigned long long>::Make(num_photon, 2)

**/

#include "scuda.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SSEQ_METHOD __device__ __forceinline__
#else
#    define SSEQ_METHOD inline 
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#define FFS(x)   (__ffs(x))
#define FFSLL(x) (__ffsll(x))
#else
#define FFS(x)   (ffs(x))
#define FFSLL(x) (ffsll(x))
#endif


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#endif

struct sseq
{
    unsigned long long seqhis ; 
    unsigned long long seqbnd ; 

    SSEQ_METHOD void zero() { seqhis = 0ull ; seqbnd = 0ull ; }
    SSEQ_METHOD void add_nibble( unsigned bounce, unsigned flag, unsigned boundary ); 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    SSEQ_METHOD std::string desc() const ; 
#endif
};

/**
sseq::add_nibble
------------------

Populates one nibble of the seqhis+seqbnd bitfields 

Hmm signing the boundary for each step would eat into bits too much, perhaps 
just collect material, as done in old workflow ?

**/

SSEQ_METHOD void sseq::add_nibble(unsigned bounce, unsigned flag, unsigned boundary )
{
    seqhis |=  (( FFS(flag) & 0xfull ) << 4*bounce ); 
    seqbnd |=  (( boundary  & 0xfull ) << 4*bounce ); 
    // 0xfull is needed to avoid all bits above 32 getting set
    // NB: nibble restriction of each "slot" means there is absolute no need for FFSLL
}

#if defined(__CUDACC__) || defined(__CUDABE__)
#else

SSEQ_METHOD std::string sseq::desc() const 
{
    std::stringstream ss ; 
    ss 
         << " seqhis " << std::setw(16) << std::hex << seqhis << std::dec 
         << " seqbnd " << std::setw(16) << std::hex << seqbnd << std::dec 
         ;
    std::string s = ss.str(); 
    return s ; 
}
#endif


