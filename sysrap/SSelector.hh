#pragma once

#include "SYSRAP_API_EXPORT.hh"

template <unsigned N> struct SEnabled ;  

struct SYSRAP_API SSelector
{
    SSelector(); 

    bool isCompoundEnabled( unsigned mmIdx ); 
    bool isShapeEnabled(    unsigned lvIdx ); 

    SEnabled<64>*  emm ; 
    SEnabled<512>* elv ; 
};



