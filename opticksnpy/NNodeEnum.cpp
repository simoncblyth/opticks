
#include <cstddef>
#include <cmath>
#include <sstream>

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


const char* NNodeEnum::POINT_INSIDE_ = "POINT_INSIDE" ;
const char* NNodeEnum::POINT_SURFACE_ = "POINT_SURFACE" ;
const char* NNodeEnum::POINT_OUTSIDE_ = "POINT_OUTSIDE" ;

const char* NNodeEnum::PointType(NNodePointType pt)
{
    const char* s = NULL ;
    switch(pt)
    {
        case POINT_INSIDE: s = POINT_INSIDE_ ; break ; 
        case POINT_SURFACE: s = POINT_SURFACE_ ; break ; 
        case POINT_OUTSIDE: s = POINT_OUTSIDE_ ; break ; 
    }
    return s ;
}

NNodePointType NNodeEnum::PointClassify( float sd, float epsilon )
{
    return fabsf(sd) < epsilon ? POINT_SURFACE : ( sd < 0 ? POINT_INSIDE : POINT_OUTSIDE ) ; 
}

std::string NNodeEnum::PointMask(unsigned mask)
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < 3 ; i++) 
    {
        NNodePointType pt = (NNodePointType)(0x1 << i) ;
        if( pt & mask ) ss << PointType(pt) << " " ;  
    }
    return ss.str();
}



