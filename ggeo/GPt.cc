#include <cstring>
#include <sstream>
#include <iomanip>

#include "GPt.hh"
#include "GLMFormat.hpp"


GPt::GPt( int lvIdx_, int ndIdx_, const char* spec_, const glm::mat4& placement_ )
    :
    lvIdx(lvIdx_),
    ndIdx(ndIdx_),
    spec(strdup(spec_)),
    placement(placement_)
{
} 

GPt::GPt( int lvIdx_, int ndIdx_, const char* spec_ )
    :
    lvIdx(lvIdx_),
    ndIdx(ndIdx_),
    spec(strdup(spec_)),
    placement(1.0f)
{
} 

void GPt::setPlacement( const glm::mat4& placement_ )
{
    placement = placement_ ;  
}


std::string GPt::desc() const 
{
    std::stringstream ss ; 
    ss 
       << " lvIdx " << std::setw(4) << lvIdx
       << " ndIdx " << std::setw(7) << ndIdx
       << " spec " << std::setw(30) << spec
       << " placement " << gformat( placement )   
       ; 

    return ss.str(); 
}

 
