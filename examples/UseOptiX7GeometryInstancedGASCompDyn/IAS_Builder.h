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
    static IAS Build( const std::vector<float>& tr );
    static IAS Build( const std::vector<glm::mat4>& tr );
    static IAS Build( const float* vals, unsigned num_vals );
    static IAS Build( const std::vector<OptixInstance>& instances);
};


