#pragma once

#include <vector>
#include "AS.h"
#include "BI.h"

struct Shape ; 

struct GAS : public AS
{
    const Shape*    sh ; 
    std::vector<BI> bis ; 
};



