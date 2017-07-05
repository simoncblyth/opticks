#include <iostream>
#include <sstream>
#include <iomanip>
#include <limits>

#include "NBBox.hpp"
#include "NGLM.hpp"
#include <glm/gtx/component_wise.hpp>

#include "GVector.hh"
#include "GBBox.hh"


gbbox::gbbox(const nbbox& other )
{
    min.x = other.min.x ; 
    min.y = other.min.y ; 
    min.z = other.min.z ;
 
    max.x = other.max.x ; 
    max.y = other.max.y ; 
    max.z = other.max.z ;
}
   

gfloat3 gbbox::dimensions()
{
    return gfloat3(max.x - min.x, max.y - min.y, max.z - min.z );
} 

gfloat3 gbbox::center()
{
   return gfloat3( (max.x + min.x)/2.0f , (max.y + min.y)/2.0f , (max.z + min.z)/2.0f ) ;
}

void gbbox::enlarge(float factor)  //  multiple of extent
{
   gfloat3 dim = dimensions();
   float ext = extent(dim); 
   float amount = ext*factor ; 
   min -= gfloat3(amount) ;
   max += gfloat3(amount) ;
}

void gbbox::include(const gbbox& other)
{ 
   min = gfloat3::minimum( min, other.min ); 
   max = gfloat3::maximum( max, other.max ); 
}

gbbox& gbbox::operator *= (const GMatrixF& m)
{
   min *= m ; 
   max *= m ; 
   return *this ;
   // hmm rotations will make the new bbox not axis aligned... see nbbox::transform
}

float gbbox::extent(const gfloat3& dim)
{
   float _extent(0.f) ;
   _extent = std::max( dim.x , _extent );
   _extent = std::max( dim.y , _extent );
   _extent = std::max( dim.z , _extent );
   _extent = _extent / 2.0f ;         
   return _extent ; 
}

gfloat4 gbbox::center_extent()
{
   gfloat3 cen = center();
   gfloat3 dim = dimensions();
   float ext = extent(dim); 
   return gfloat4( cen.x, cen.y, cen.z, ext );
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


// hmm not good to mix lingo ... 
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




