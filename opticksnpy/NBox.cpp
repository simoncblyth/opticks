
#include "NGLMExt.hpp"
#include <glm/gtx/component_wise.hpp>

#include "NBBox.hpp"
#include "NBox.hpp"
#include "NPart.hpp"
#include "NPlane.hpp"

#include <cmath>
#include <cassert>
#include <cstring>

#include "OpticksCSG.h"

/**
~/opticks_refs/Procedural_Modelling_with_Signed_Distance_Functions_Thesis.pdf

SDF from point px,py,pz to box at origin with side lengths (sx,sy,sz) at the origin 

    max( abs(px) - sx/2, abs(py) - sy/2, abs(pz) - sz/2 )

**/

float nbox::operator()(float x, float y, float z) 
{
    glm::vec4 q(x,y,z,1.0); 
    if(gtransform) q = gtransform->v * q ;

    glm::vec3 p = glm::vec3(q) - center ;  // coordinates in frame with origin at box center 
    glm::vec3 a = glm::abs(p) ;
    glm::vec3 s( param.w );      
    glm::vec3 d = a - s ; 

    return glm::compMax(d) ;
} 

float nbox::sdf1(float x, float y, float z)
{
    return (*this)(x,y,z);
}

float nbox::sdf2(float x, float y, float z)
{
    glm::vec4 p(x,y,z,1.0); 
    if(gtransform) p = gtransform->v * p ;

    glm::vec3 bmax = center + glm::vec3( param.w ) ; // 

    glm::vec3 d = glm::abs(glm::vec3(p)) - bmax  ;

    float dmaxcomp = glm::compMax(d);

    glm::vec3 dmax = glm::max( d, glm::vec3(0.f) );

    float d_inside = fminf(dmaxcomp, 0.f);
    float d_outside = glm::length( dmax );

    return d_inside + d_outside ;       

   // see tests/NBoxTest.cc   sdf2 and sdf1 match despite code appearances
}




nbbox nbox::bbox()
{
    nbbox bb ;

    float s  = param.w ; 
    bb.min = make_nvec3( param.x - s, param.y - s, param.z - s );
    bb.max = make_nvec3( param.x + s, param.y + s, param.z + s );
    bb.side = bb.max - bb.min ; 

    return gtransform ? bb.transform(gtransform->t) : bb ; 

    // bbox transforms need TR not IR*IT as they apply directly to geometry 
    // unlike transforming the SDF point or ray tracing ray which needs the inverse irit 
}

glm::vec3 nbox::gcenter()
{
    return gtransform == NULL ? center : glm::vec3( gtransform->t * glm::vec4(center, 1.f ) ) ;
}

void nbox::pdump(const char* msg, int verbosity )
{
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "no-label" )
              << " center " << center 
              << " side " << param.w 
              << " gcenter " << gcenter()
              << " gtransform? " << !!gtransform
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}




