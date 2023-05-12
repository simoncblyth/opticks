#pragma once

/**
squadx.h : facilitate storing uint64_t values into quad4 which uses 32 bit elements
=======================================================================================


**/

#include <vector_types.h>
#include <cstdint>

struct wuint2 {  uint64_t x, y ; };

union quadx
{
    float4 f ; 
    int4   i ; 
    uint4  u ; 
    wuint2 w ; 
};


struct quadx4 
{ 
    quadx q0 ; 
    quadx q1 ; 
    quadx q2 ; 
    quadx q3 ;
}; 




