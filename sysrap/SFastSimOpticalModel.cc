#include <cstring>
#include "SFastSimOpticalModel.hh"

const SFastSimOpticalModel* SFastSimOpticalModel::INSTANCE = nullptr ; 
const SFastSimOpticalModel* SFastSimOpticalModel::Get(){ return INSTANCE ; }
char SFastSimOpticalModel::GetStatus(){  return INSTANCE ? INSTANCE->getStatus() : 'X' ; }

SFastSimOpticalModel::SFastSimOpticalModel(const char* name_)
    :
    name( name_ ? strdup(name_) : nullptr )
{
    INSTANCE = this ; 
}
 
