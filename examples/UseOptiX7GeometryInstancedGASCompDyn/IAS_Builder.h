#pragma once

#include <optix.h>
#include <vector>

#include "GAS.h"
#include "IAS.h"

struct Geo ; 
struct Grid ; 

struct IAS_Builder
{
    static void Build( IAS& ias, const Grid* gr, const std::vector<GAS>& vgas  );  
    static void Build( IAS& ias, const std::vector<OptixInstance>& instances);
};


