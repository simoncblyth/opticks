#include <cstring>
#include "SFastSimOpticalModel.hh"

std::vector<SFastSimOpticalModel*> SFastSimOpticalModel::record = {} ;

SFastSimOpticalModel::SFastSimOpticalModel(const char* name_)
    :
    name( name_ ? strdup(name_) : nullptr )
{
    record.push_back(this); 
}
 
