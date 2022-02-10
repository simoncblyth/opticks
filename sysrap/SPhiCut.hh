#pragma once
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

union quad ; 

struct SYSRAP_API SPhiCut
{
    static const plog::Severity LEVEL ; 
    static void PrepareParam( quad& q0, double startPhi_pi, double deltaPhi_pi );  
};

