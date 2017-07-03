#include <iostream>
#include <sstream>
#include <iomanip>
#include <limits>

#include "NGLM.hpp"
#include <glm/gtx/component_wise.hpp>

#include "GVector.hh"


std::string gfloat3::desc() const 
{
    std::stringstream ss ; 
    ss
          << "(" << std::setw(10) << std::fixed << std::setprecision(3) << x 
          << " " << std::setw(10) << std::fixed << std::setprecision(3) << y 
          << " " << std::setw(10) << std::fixed << std::setprecision(3) << z
          << ")"
          ; 

    return ss.str(); 
}


std::string gfloat4::desc() const 
{
    std::stringstream ss ; 
    ss
          << "(" << std::setw(10) << std::fixed << std::setprecision(3) << x 
          << " " << std::setw(10) << std::fixed << std::setprecision(3) << y 
          << " " << std::setw(10) << std::fixed << std::setprecision(3) << z
          << " " << std::setw(10) << std::fixed << std::setprecision(3) << w
          << ")"
          ; 

    return ss.str(); 
}


void gbbox::Summary(const char* msg) const 
{
    std::cout << msg << desc() << std::endl ; 
}

std::string gbbox::description() const 
{
    return desc();
}

std::string gbbox::desc() const 
{
    std::stringstream ss ; 
    ss 
       << " mn " << min.desc()
       << " mx " << max.desc()
        ;
    return ss.str(); 
}


float gbbox::MaxDiff( const gbbox& a, const gbbox& b)
{
    glm::vec3 amn(a.min.x, a.min.y, a.min.z);
    glm::vec3 amx(a.max.x, a.max.y, a.max.z);

    glm::vec3 bmn(b.min.x, b.min.y, b.min.z);
    glm::vec3 bmx(b.max.x, b.max.y, b.max.z);

    glm::vec3 dmn = glm::abs(amn - bmn) ;
    glm::vec3 dmx = glm::abs(amx - bmx) ;

    return std::max<float>( glm::compMax(dmn), glm::compMax(dmx) );
}









std::string guint4::description() const 
{
    std::stringstream ss ; 

    //char s[64] ;
    //snprintf(s, 64, " (%3u,%3u,%3u,%3u) ", x, y, z, w);
    //return s ; 

    unsigned umax = std::numeric_limits<unsigned>::max() ;

    ss << " (" << std::setw(3) << x << "," ;

    if(y == umax)
        ss << " - " ;
    else
        ss << std::setw(3) << y  ;
            
    ss << "," ;

    if(z == umax)
        ss << " - " ;
    else
        ss << std::setw(3) << z  ;
 
    ss << "," << std::setw(3) << w << ") " ; 

    return ss.str(); 
}





