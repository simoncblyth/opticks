#pragma once
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

union quad ; 

struct SYSRAP_API SThetaCut
{
    static const plog::Severity LEVEL ; 
    static void PrepareParam( quad& q0, quad& q1, double startTheta_pi, double deltaTheta_pi );  

};

