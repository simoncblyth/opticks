#pragma once
#include "SYSRAP_API_EXPORT.hh"


/**
SSeq : integers as sequences of nibbles (4-bits)
==================================================

**/


template <typename T>
struct SYSRAP_API SSeq
{
    SSeq(T seq_) ;

    T msn();   // most significant nibble
    T nibble(unsigned i);

    T seq ; 
    T zero ;     

};





