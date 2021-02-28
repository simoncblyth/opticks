#pragma once

#include <optix.h>
#include "IAS.h"

struct Grid ; 
struct SBT ; 

struct IAS_Builder
{
    static void Build( IAS& ias, const Grid* gr, const SBT* sbt  );  
    static void Build( IAS& ias, const std::vector<OptixInstance>& instances);
};


