#include "IAS.h"
#include "IAS_Builder.h"

IAS IAS::Build(const std::vector<float>& tr )
{
    return IAS_Builder::Build(tr); 
}
IAS IAS::Build(const std::vector<glm::mat4>& tr )
{
    return IAS_Builder::Build(tr); 
}
IAS IAS::Build(const float* vals, unsigned num_val )
{
    return IAS_Builder::Build(vals, num_val); 
}



