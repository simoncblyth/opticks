
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
    glm::vec4 p1 = gtransform ? gtransform->irit * p0 : p0 ; 

    /*
    if(gtransform)
        std::cout << "nbox::operator"
                  << " p0 " << p0 
                  << " p1 " << p1
                  << " gtransform "  << *gtransform 
                  << std::endl ;  
    */

    glm::vec3 pc = glm::vec3(p1) - center ;  // coordinates in frame with origin at box center 
    glm::vec3 a = glm::abs(pc) ;
    glm::vec3 s( param.w );
    glm::vec3 d = a - s ; 

    return gmaxf(d) ;
} 

/*
    nvec3 p = make_nvec3( px - param.x, py - param.y, pz - param.z ); // in the frame of the box
    nvec3 a = nabsf(p) ; 
    nvec3 s = make_nvec3( param.w, param.w, param.w );          
    nvec3 d = a - s ; 
    return nmaxf(d) ;
*/



nbbox nbox::bbox()
{
    nbbox bb ;

    float s  = param.w ; 
    bb.min = make_nvec3( param.x - s, param.y - s, param.z - s );
    bb.max = make_nvec3( param.x + s, param.y + s, param.z + s );
    bb.side = bb.max - bb.min ; 

    return gtransform ? bb.transform(gtransform->tr) : bb ; 

    // bbox transforms need TR not IR*IT as they apply directly to geometry 
    // unlike transforming the SDF point or ray tracing ray which needs the inverse irit 
}


