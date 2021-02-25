#pragma once

#include <vector>
#include "AS.h"
#include "BI.h"

struct GAS : public AS
{
    std::vector<float>      extents ; 
    unsigned                num_sbt_rec ; 
    std::vector<BI>         bis ; 
};



