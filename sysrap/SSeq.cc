

#include <cassert>
#include "SSeq.hh"


template <typename T>
SSeq<T>::SSeq(T seq_) 
   : 
   seq(seq_), 
   zero(0ull) 
{} ;

template <typename T>
T SSeq<T>::msn()   // most significant nibble
{
    unsigned nnib = sizeof(T)*2 ; 
    for(unsigned i=0 ; i < nnib ; i++)
    {
        T f = nibble(nnib-1-i) ; 
        if( f == zero ) continue ; 
        return f ; 
    } 
    return zero ; 
}


template <typename T>
T SSeq<T>::nibble(unsigned i)
{
    return (seq >> i*4) & T(0xF) ; 
}


template struct SSeq<unsigned>;
template struct SSeq<unsigned long long>;


