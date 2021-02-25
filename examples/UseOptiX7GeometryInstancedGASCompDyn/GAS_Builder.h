#pragma once

#include <vector>
#include "GAS.h"
#include "BI.h"

struct GAS_Builder
{
    static BI MakeCustomPrimitivesBI(const float* bb, unsigned num_val, unsigned primitiveIndexOffset );

    static void Build(GAS& gas, const std::vector<float>& bb); 
    static void Build(GAS& gas);
};


