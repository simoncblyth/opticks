#pragma once

#include <vector>
#include "GAS.h"

struct GAS_Builder
{
    static GAS Build(const std::vector<float>& bb); 
    static GAS Build(OptixBuildInput aabb_input); 
};


