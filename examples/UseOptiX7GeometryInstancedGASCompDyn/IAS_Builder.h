#pragma once

#include <optix.h>
#include <vector>
#include <glm/glm.hpp>

#include "IAS.h"

/**
IAS_Builder
===========

**/


struct IAS_Builder
{
    static void Build( IAS& ias );  // must has ias.trs set already 
    static void Build( IAS& ias, const std::vector<OptixInstance>& instances);
};


