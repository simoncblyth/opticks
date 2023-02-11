#pragma once
/**
sxf.h : simple wrapper to give uniform behaviour to spa/sxf/sbb
===================================================================

**/

#include <string>
#include <sstream>
#include <iomanip>

#include <glm/glm.hpp>
#include "glm/gtx/string_cast.hpp"

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API sxf
{
    static constexpr const char* NAME = "sxf" ; 
    glm::tmat4x4<double> mat ; 
    std::string desc() const ;  
}; 

inline std::string sxf::desc() const 
{
    std::stringstream ss ;
    ss << glm::to_string(mat) ; 
    std::string str = ss.str(); 
    return str ; 
}


