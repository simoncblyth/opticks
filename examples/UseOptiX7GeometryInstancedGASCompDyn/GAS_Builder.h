#pragma once

#include <vector>
#include "GAS.h"
#include "BI.h"

struct Shape ; 

struct GAS_Builder
{
    static BI MakeCustomPrimitivesBI(const Shape* sh, unsigned i ); 
    static void Build(GAS& gas, const Shape* sh  ); 
    static void Build(GAS& gas);
};


