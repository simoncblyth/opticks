#pragma once 

#include <string>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SFrameConfig 
{
    static constexpr const char* kFrameMask   = "OPTICKS_FRAME_MASK" ; 
    static constexpr const char* _FrameMaskAll = "pixel,isect,fphoton"  ; 
    static constexpr const char* _FrameMaskDefault = "pixel"  ; 
    static unsigned _FrameMask ; 

    static unsigned FrameMask(); 
    static void SetFrameMask(const char* names, char delim=',') ; 
    static std::string FrameMaskLabel(); 

    static std::string Desc(); 
}; 



