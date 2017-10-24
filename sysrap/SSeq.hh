#pragma once
#include "SYSRAP_API_EXPORT.hh"



template <typename T>
struct SYSRAP_API SSeq
{
    SSeq(T seq_) ;

    T msn();   // most significant nibble
    T nibble(unsigned i);

    T seq ; 
    T zero ;     

};





