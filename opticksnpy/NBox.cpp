
#include "NGLMExt.hpp"

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

double nbox::operator()(double px, double py, double pz) 
{
    glm::vec4 p0(px,py,pz,1.0); 
    glm::vec4 p1 = gtransform ? gtransform->v * p0 : p0 ; 

    glm::vec3 pc = glm::vec3(p1) - center ;  // coordinates in frame with origin at box center 
    glm::vec3 a = glm::abs(pc) ;
    glm::vec3 s( param.w );
    glm::vec3 d = a - s ; 

    return gmaxf(d) ;
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




