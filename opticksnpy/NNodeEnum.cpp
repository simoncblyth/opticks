
#include <cstddef>
#include "NNodeEnum.hpp"


const char* NNodeEnum::FRAME_MODEL_ = "FRAME_MODEL" ;
const char* NNodeEnum::FRAME_LOCAL_ = "FRAME_LOCAL" ;
const char* NNodeEnum::FRAME_GLOBAL_ = "FRAME_GLOBAL" ;

const char* NNodeEnum::FrameType(NNodeFrameType fr)
{
    const char* s = NULL ;
    switch(fr)
    {
        case FRAME_MODEL: s = FRAME_MODEL_ ; break ; 
        case FRAME_LOCAL: s = FRAME_LOCAL_ ; break ; 
        case FRAME_GLOBAL: s = FRAME_GLOBAL_ ; break ; 
    }
    return s ;
}


