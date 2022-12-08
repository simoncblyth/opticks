#pragma once
/**
sseq.h : photon level step-by-step history and material recording seqhis/seqmat using NSEQ 64 bit uint
========================================================================================================

Note that needing NSEQ=2 (to allow storing 32 flags) follows from SEventConfig::MaxBounce 
being 32 (rather than the long held 16).

Could consider changing NSEQ based on config somehow, but that is not worthy of 
the complexity as sseq recording is a debugging activity not a production activity. 
So memory usage and performance of sseq is not important, as it will not 
be used in production running by definition. 

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

#define rFFS(x)   ( x == 0 ? 0 : 0x1 << (x - 1) ) 
#define rFFSLL(x) ( x == 0 ? 0 : 0x1ull << (x - 1) )   
// use rFFSLL when values of x exceed about 31


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "smath.h"
#include "OpticksPhoton.hh"
#include "sstr.h"
#endif



struct sseq
{
    static constexpr const unsigned NSEQ = 2 ;  
    static constexpr const unsigned BITS = 4 ;
    static constexpr const unsigned long long MASK = ( 0x1ull << BITS ) - 1ull ;
    static constexpr const unsigned SLOTMAX = 64/BITS ; 
    static constexpr const unsigned SLOTS = SLOTMAX*NSEQ ;

    typedef unsigned long long ULL ; 
    ULL seqhis[NSEQ] ; 
    ULL seqbnd[NSEQ] ; 

    static SSEQ_METHOD unsigned GetNibble(const ULL& seq, unsigned slot) ;  
    static SSEQ_METHOD void     ClearNibble(    ULL& seq, unsigned slot) ;  
    static SSEQ_METHOD void     SetNibble(      ULL& seq, unsigned slot, unsigned value ) ;  

    SSEQ_METHOD unsigned get_flag(unsigned slot) const ;
    SSEQ_METHOD void     set_flag(unsigned slot, unsigned flag) ;

    SSEQ_METHOD int      seqhis_nibbles() const ;
    SSEQ_METHOD int      seqbnd_nibbles() const ;

    SSEQ_METHOD void zero(); 
    SSEQ_METHOD void add_nibble( unsigned slot, unsigned flag, unsigned boundary ); 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    SSEQ_METHOD std::string desc() const ; 
    SSEQ_METHOD std::string desc_seqhis() const ; 
    SSEQ_METHOD std::string brief() const ; 
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

/*
SSEQ_METHOD unsigned sseq::get_flag(unsigned slot) const 
{
    unsigned f = GetNibble(seqhis, slot) ; 
    return  f == 0 ? 0 : 0x1ull << (f - 1) ; 
}
SSEQ_METHOD void sseq::set_flag(unsigned slot, unsigned flag)
{
    SetNibble(seqhis, slot, FFS(flag)) ; 
}
*/

SSEQ_METHOD unsigned sseq::get_flag(unsigned slot) const 
{
    unsigned iseq = slot/SLOTMAX ; 
    unsigned f = iseq < NSEQ ? ( seqhis[iseq] >> BITS*(slot - iseq*SLOTMAX) ) & MASK : 0  ; 
    return  f == 0 ? 0 : 0x1ull << (f - 1) ;   // reconstitute flag from bitpos, inverse of FFS
}
SSEQ_METHOD void sseq::set_flag(unsigned slot, unsigned flag)
{
    unsigned iseq = slot/SLOTMAX ;  // iseq:element to write to 
    if(iseq < NSEQ) seqhis[iseq] |=  (( FFS(flag) & MASK ) << BITS*(slot - iseq*SLOTMAX) );

    // NB: note that this does not clear first, so it requires starting with zeroed elements 
    //
    // slot ranges across all elements, so subtracting total slots for preceding elements (iseq*SLOTMAX)
    // gives the appropriate shift for the iseq element 
}

SSEQ_METHOD int sseq::seqhis_nibbles() const 
{ 
    int count = 0 ;
    for(unsigned i=0 ; i < NSEQ ; i++) count += smath::count_nibbles(seqhis[i]) ;
    return count ; 
}
SSEQ_METHOD int sseq::seqbnd_nibbles() const 
{
    int count = 0 ;
    for(unsigned i=0 ; i < NSEQ ; i++) count += smath::count_nibbles(seqbnd[i]) ;
    return count ; 
}

SSEQ_METHOD void sseq::zero()
{ 
    for(unsigned i=0 ; i < NSEQ ; i++)
    { 
        seqhis[i] = 0ull ; 
        seqbnd[i] = 0ull ; 
    }
}

/**
sseq::add_nibble
------------------

Populates one nibble of the seqhis+seqbnd bitfields 

Hmm signing the boundary for each step would eat into bits too much, perhaps 
just collect material, as done in old workflow ?

Have observed (see test_desc_seqhis) that when adding more than 16 nibbles 
into the 64 bit ULL (which will not fit), get unexpected "mixed" wraparound 
not just simply overwriting.  

0xfull is needed to avoid all bits above 32 getting set
NB: nibble restriction of each "slot" means there is absolute no need for FFSLL

**/

SSEQ_METHOD void sseq::add_nibble(unsigned slot, unsigned flag, unsigned boundary )
{
    unsigned iseq = slot/SLOTMAX ;  // iseq:element to write to 
    unsigned shift = BITS*(slot - iseq*SLOTMAX) ; 

    if(iseq < NSEQ) 
    {  
        seqhis[iseq] |=  (( FFS(flag) & MASK ) << shift ); 
        seqbnd[iseq] |=  (( boundary  & MASK ) << shift ); 
    }
}

#if defined(__CUDACC__) || defined(__CUDABE__)
#else

SSEQ_METHOD std::string sseq::desc() const 
{
    std::stringstream ss ; 
    ss << " seqhis " ; 
    for(unsigned i=0 ; i < NSEQ ; i++) ss << " " << std::setw(16) << std::hex << seqhis[NSEQ-1-i] << std::dec  ; 
    ss << " seqbnd " ; 
    for(unsigned i=0 ; i < NSEQ ; i++) ss << " " << std::setw(16) << std::hex << seqhis[NSEQ-1-i] << std::dec  ; 
    std::string s = ss.str(); 
    return s ; 
}

SSEQ_METHOD std::string sseq::desc_seqhis() const 
{
    std::string fseq = OpticksPhoton::FlagSequence(&seqhis[0], NSEQ) ;  // HMM: bring this within sseq ? 

    std::stringstream ss ; 
    for(unsigned i=0 ; i < NSEQ ; i++) ss << " " << std::setw(16) << std::hex << seqhis[NSEQ-1-i] << std::dec  ; 
    ss << " nib " << std::setw(2) << seqhis_nibbles() 
       << " " << sstr::TrimTrailing(fseq.c_str()) 
       ;
    std::string s = ss.str(); 
    return s ; 
}
SSEQ_METHOD std::string sseq::brief() const
{
    std::string fseq = OpticksPhoton::FlagSequence(&seqhis[0], NSEQ) ;  // HMM: bring this within sseq ? 
    return sstr::TrimTrailing(fseq.c_str()) ; 
} 

#endif


