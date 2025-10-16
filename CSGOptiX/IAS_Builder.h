#pragma once

#include <optix.h>
#include "IAS.h"
#include "sqat4.h"

struct SBT ; 

struct IAS_Builder
{
    static void Build_OLD( IAS& ias, const std::vector<OptixInstance>& instances );
};


