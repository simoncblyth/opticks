#pragma once

#include <vector>
#include "GAS.h"
#include "BI.h"

struct GAS_Builder
{
    static GAS Build(const std::vector<float>& bb); 
    static BI MakeCustomPrimitivesBI(const float* bb, unsigned num_val, unsigned primitiveIndexOffset );
    static GAS Build(const std::vector<BI>& bis);
};


