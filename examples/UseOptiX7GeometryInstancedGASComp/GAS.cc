#include "GAS.h"
#include "GAS_Builder.h"

GAS GAS::Build(const std::vector<float>& bb )
{
    return GAS_Builder::Build(bb); 
}

