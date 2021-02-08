#pragma once

#include <vector>
#include "AS.h"

struct GAS : public AS
{
    static GAS Build(const std::vector<float>& bb ); // must contain a multiple of 6 min/max bbox floats  
};



