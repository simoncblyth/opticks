
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


float nzsphere::operator()(float x, float y, float z) const 
{
    glm::vec4 p(x,y,z,1.f); 
    if(gtransform) p = gtransform->v * p ;  // v:inverse-transform

    float d_sphere = glm::distance( glm::vec3(p), center ) - radius ;

    float d_slab = fmaxf( p.z - zmax(), -(p.z - zmin()) );  

    float sd = fmaxf( d_sphere, d_slab );  // CSG intersect of sphere with slab

    return complement ? -sd : sd ; 
} 

nbbox nzsphere::bbox() const 
{
    nbbox bb = make_bbox();
    assert( zmax() > zmin() ); 
    bb.max = make_nvec3(center.x + radius, center.y + radius, zmax() );
    bb.min = make_nvec3(center.x - radius, center.y - radius, zmin() );
    bb.side = bb.max - bb.min ; 

    return gtransform ? bb.transform(gtransform->t) : bb ; 
}

glm::vec3 nzsphere::gseedcenter()
{
    glm::vec4 seedcenter( center.x, center.y, (zmin() + zmax())/2.f, 1.f ); 
    return apply_gtransform(seedcenter);
    //if(gtransform) seedcenter = gtransform->t * seedcenter ; 
    //return glm::vec3(seedcenter) ; 
}

glm::vec3 nzsphere::gseeddir()
{
    glm::vec4 dir(0,0,1,0); 
    return apply_gtransform(dir);
    //if(gtransform) dir = gtransform->t * dir ; 
    //return glm::vec3(dir) ;
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
              << " gseedcenter " << gseedcenter()
              << " gtransform " << !!gtransform 
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}

