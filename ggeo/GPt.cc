#include <cstring>
#include <sstream>
#include <iomanip>

#include "GPt.hh"
#include "GLMFormat.hpp"


GPt::GPt( unsigned tree_, const char* spec_, const glm::mat4& placement_ )
    :
    tree(tree_),
    spec(strdup(spec_)),
    placement(placement_)
{
} 

GPt::GPt( unsigned tree_, const char* spec_ )
    :
    tree(tree_),
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
       << " tree " << std::setw(4) << tree
       << " spec " << std::setw(30) << spec
       << " placement " << gformat( placement )   
       ; 

    return ss.str(); 
}

 
