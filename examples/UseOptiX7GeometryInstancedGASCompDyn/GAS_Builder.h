#pragma once

#include <vector>
#include "GAS.h"
#include "BI.h"

struct Shape ; 

struct GAS_Builder
{
    static BI MakeCustomPrimitivesBI(const float* bb, unsigned num_bb_val,  const float* param, unsigned num_param_val, unsigned primitiveIndexOffset ); 

    static void Build(GAS& gas, const Shape* sh  ); 
    static void Build(GAS& gas);
};


