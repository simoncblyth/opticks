#pragma once

#include <string>
#include "NGLM.hpp"
#include "GGEO_HEAD.hh"
#include "GGEO_API_EXPORT.hh"
 
struct GGEO_API GPt 
{
    unsigned    tree ; 
    std::string spec ; 
    glm::mat4   placement ; 

    GPt( unsigned tree_, const char* spec_, const glm::mat4& placement_ ); 
    GPt( unsigned tree_, const char* spec_ ); 
    void setPlacement( const glm::mat4& placement_ ); 

    std::string desc() const ; 

}; 

#include "GGEO_TAIL.hh"

