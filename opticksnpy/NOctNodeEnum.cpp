#include <cstddef>
#include "NOctNodeEnum.hpp"

const char* NOctNodeEnum::NOCT_ZERO_     = "ZERO" ;
const char* NOctNodeEnum::NOCT_INTERNAL_ = "INTERNAL" ;
const char* NOctNodeEnum::NOCT_LEAF_     = "LEAF" ;

const char* NOctNodeEnum::NOCTName(NOctNode_t type)
{
    switch(type)
    {
       case NOCT_ZERO:     return NOCT_ZERO_     ; break ; 
       case NOCT_INTERNAL: return NOCT_INTERNAL_ ; break ; 
       case NOCT_LEAF:     return NOCT_LEAF_     ; break ; 
    }
    return NULL ; 
}




