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
    typedef unsigned long long ULL ; 
    static SSEQ_METHOD unsigned GetNibble(const ULL& seq, unsigned slot) ;  
    static SSEQ_METHOD void     ClearNibble(    ULL& seq, unsigned slot) ;  
    static SSEQ_METHOD void     SetNibble(      ULL& seq, unsigned slot, unsigned value ) ;  

    ULL seqhis ; 
    ULL seqbnd ; 

    SSEQ_METHOD void zero() { seqhis = 0ull ; seqbnd = 0ull ; }
    SSEQ_METHOD void add_nibble( unsigned slot, unsigned flag, unsigned boundary ); 

    SSEQ_METHOD unsigned get_flag(unsigned slot) const ;
    SSEQ_METHOD void     set_flag(unsigned slot, unsigned flag) ;

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    SSEQ_METHOD std::string desc() const ; 
#endif
};


SSEQ_METHOD unsigned sseq::GetNibble(const unsigned long long& seq, unsigned slot)
{ 
    return ( seq >> 4*slot ) & 0xfull ; 
}

SSEQ_METHOD void sseq::ClearNibble(unsigned long long& seq, unsigned slot)
{
    seq &= ~( 0xfull << 4*slot ) ; 
}

SSEQ_METHOD void sseq::SetNibble(unsigned long long& seq, unsigned slot, unsigned value)
{ 
    seq =  ( seq & ~( 0xfull << 4*slot )) | ( (value & 0xfull) << 4*slot ) ;   
}

SSEQ_METHOD unsigned sseq::get_flag(unsigned slot) const 
{
    unsigned f = GetNibble(seqhis, slot) ; 
    return  f == 0 ? 0 : 0x1 << (f - 1) ; 
}
SSEQ_METHOD void sseq::set_flag(unsigned slot, unsigned flag)
{
    SetNibble(seqhis, slot, FFS(flag)) ; 
}




/**
sseq::add_nibble
------------------

Populates one nibble of the seqhis+seqbnd bitfields 

Hmm signing the boundary for each step would eat into bits too much, perhaps 
just collect material, as done in old workflow ?

**/

SSEQ_METHOD void sseq::add_nibble(unsigned slot, unsigned flag, unsigned boundary )
{
    seqhis |=  (( FFS(flag) & 0xfull ) << 4*slot ); 
    seqbnd |=  (( boundary  & 0xfull ) << 4*slot ); 
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


