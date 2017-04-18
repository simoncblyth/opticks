
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>

#include "NGLMExt.hpp"

// sysrap-
#include "OpticksCSG.h"

// npy-
#include "NZSphere.hpp"
#include "NBBox.hpp"

#include "PLOG.hh"


float nzsphere::operator()(float x, float y, float z) 
{
    glm::vec4 p(x,y,z,1.f); 
    if(gtransform) p = gtransform->v * p ;  // v:inverse-transform

    float d_sphere = glm::distance( glm::vec3(p), center ) - radius ;

    float d_slab = fmaxf( p.z - zmax(), -(p.z - zmin()) );  

    return fmaxf( d_sphere, d_slab );  // CSG intersect of sphere with slab
} 

nbbox nzsphere::bbox()
{
    nbbox bb = make_bbox();
    assert( zmax() > zmin() ); 
    bb.max = make_nvec3(center.x + radius, center.y + radius, zmax() );
    bb.min = make_nvec3(center.x - radius, center.y - radius, zmin() );
    bb.side = bb.max - bb.min ; 

    return gtransform ? bb.transform(gtransform->t) : bb ; 
}

glm::vec3 nzsphere::gcenter()
{
    return gtransform == NULL ? center : glm::vec3( gtransform->t * glm::vec4(center, 1.f ) ) ; // t:transform
}

void nzsphere::pdump(const char* msg, int verbosity)
{
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "no-label" )
              << " center " << center 
              << " radius " << radius 
              << " zmin " << zmin()
              << " zmax " << zmax()
              << " gcenter " << gcenter()
              << " gtransform " << !!gtransform 
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}

