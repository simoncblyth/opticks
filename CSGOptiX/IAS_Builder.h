#pragma once

#include <optix.h>
#include "IAS.h"
#include "sqat4.h"
#include "plog/Severity.h"

struct SBT ; 

struct IAS_Builder
{
    static const plog::Severity LEVEL ; 

    static void CollectInstances(std::vector<OptixInstance>& instances, const std::vector<qat4>& ias_inst, const SBT* sbt ); 
    static void Build( IAS& ias, const std::vector<qat4>& ias_inst, const SBT* sbt  );  
    static void Build( IAS& ias, const std::vector<OptixInstance>& instances );
};


