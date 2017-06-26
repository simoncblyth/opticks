
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


float nzsphere::operator()(float x_, float y_, float z_) const 
{
    glm::vec4 p(x_,y_,z_,1.f); 
    if(gtransform) p = gtransform->v * p ;  // v:inverse-transform

    glm::vec3 c = center(); 
    float r = radius();

    float d_sphere = glm::distance( glm::vec3(p), c ) - r ;

    float d_slab = fmaxf( p.z - zmax(), -(p.z - zmin()) );  

    float sd = fmaxf( d_sphere, d_slab );  // CSG intersect of sphere with slab

    return complement ? -sd : sd ; 
} 

nbbox nzsphere::bbox() const 
{
    glm::vec3 c = center(); 
    float r = radius(); 

    nbbox bb = make_bbox();
    bb.max = make_nvec3(c.x + r, c.y + r, zmax() );
    bb.min = make_nvec3(c.x - r, c.y - r, zmin() );
    bb.side = bb.max - bb.min ; 
    bb.invert = complement ; 
    bb.empty = false ; 

    return gtransform ? bb.transform(gtransform->t) : bb ; 
}

glm::vec3 nzsphere::gseedcenter() const 
{
    glm::vec3 c = center(); 
    glm::vec4 seedcenter( c.x, c.y, zc(), 1.f ); 
    return apply_gtransform(seedcenter);
}

glm::vec3 nzsphere::gseeddir() const 
{
    glm::vec4 dir(0,0,1,0); 
    return apply_gtransform(dir);
}

void nzsphere::pdump(const char* msg) const 
{
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "-" )
              << " center " << center()
              << " radius " << radius()
              << " zmin " << zmin()
              << " zmax " << zmax()
              << " z1 " << z1()
              << " z2 " << z2()
              << " zc " << zc()
              << " gseedcenter " << gseedcenter()
              << " gtransform " << !!gtransform 
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}

