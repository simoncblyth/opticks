#pragma once

/**
squadx.h : facilitate storing uint64_t values into quad4 which uses 32 bit elements
=======================================================================================

This is used for storing uint64_t timestmaps into the sctx.sup from SEvt::addProcessHitsStamp

**/

#include <vector_types.h>
#include <cstdint>

struct wuint2 {  uint64_t x, y ; };

union quadx
{
    float4 f ; 
    int4   i ; 
    uint4  u ; 
    wuint2 w ;   // wide 
};


struct quadx4 
{ 
    quadx q0 ; 
    quadx q1 ; 
    quadx q2 ; 
    quadx q3 ;
}; 

struct quadx6
{ 
    quadx q0 ; 
    quadx q1 ; 
    quadx q2 ; 
    quadx q3 ;
    quadx q4 ;
    quadx q5 ;
}; 


