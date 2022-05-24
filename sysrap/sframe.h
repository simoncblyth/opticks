#pragma once

#include "scuda.h"
#include "sqat4.h"

struct sframe
{
    float4 ce = {} ; 
    qat4   m2w ; 
    qat4   w2m ; 
}; 

inline std::ostream& operator<<(std::ostream& os, const sframe& fr)
{
    os 
       << " ce  " << fr.ce << std::endl 
       << " m2w " << fr.m2w << std::endl 
       << " w2m " << fr.w2m << std::endl 
       ;
    return os; 
}



