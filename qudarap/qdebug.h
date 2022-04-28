#pragma once
/**
qdebug.h
==========

Instanciation managed from QSim

**/

#include "scuda.h"
#include "squad.h"
#include "sphoton.h"

#include "qstate.h"
#include "qprd.h"

struct qdebug
{
    float wavelength ; 
    float cosTheta ; 
    float3 normal ; 

    qstate s ; 
    quad2  prd ; 
    sphoton  p ; 

}; 


