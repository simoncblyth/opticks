#pragma once

#include "scuda.h"
#include "squad.h"
#include "qstate.h"
#include "qprd.h"

struct qdebug
{
    float wavelength ; 
    float cosTheta ; 
    float3 normal ; 

    qstate s ; 
    qprd   prd ; 
    quad4  p ; 

}; 


