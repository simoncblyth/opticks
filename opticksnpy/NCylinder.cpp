
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>


#include "NGLMExt.hpp"

// sysrap-
#include "OpticksCSG.h"

// npy-
#include "NCylinder.hpp"
#include "NBBox.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"

#include "PLOG.hh"


float ncylinder::radius(){ return param.w ; }
float ncylinder::x(){      return param.x ; }
float ncylinder::y(){      return param.y ; }
float ncylinder::z(){      return param.z ; }

// signed distance function
double ncylinder::operator()(double px, double py, double pz) 
{
    glm::vec4 p0(px,py,pz,1.0); 
    glm::vec4 p = gtransform ? gtransform->v * p0 : p0 ; 
    p.z = 0.f ;  // <-- distance to z-axis parallel infinite cylinder : does not depend on z 
    float d = glm::distance( glm::vec3(p), center );
    return d - radius_ ;  
} 

/*

See env-;sdf-

http://aka-san.halcy.de/distance_fields_prefinal.pdf

By using CSG operations, we can now cut parts of the (infinite) cylinder 
by intersecting it and an infinite z-slab, resulting in a finite z-oriented 
cylinder with radius r and height h:

    d = max( sqrt(px^2+py^2) - r, |pz|-(h/2) )


* "max" corresponds to CSG intersection with z-slab (infinite in x and y)
   which is represented by 

    d = |pz| - (h/2)       <- d = 0 at  pz = +- h/2

*/



glm::vec3 ncylinder::gcenter()
{
    return gtransform == NULL ? center : glm::vec3( gtransform->t * glm::vec4(center, 1.f ) ) ;
}

void ncylinder::pdump(const char* msg, int verbosity)
{
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "no-label" )
              << " center " << center 
              << " radius_ " << radius_ 
              << " gcenter " << gcenter()
              << " gtransform " << !!gtransform 
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}


nbbox ncylinder::bbox()
{
    nbbox bb = make_nbbox();

    bb.min = make_nvec3(param.x - param.w, param.y - param.w, param.z - param.w);
    bb.max = make_nvec3(param.x + param.w, param.y + param.w, param.z + param.w);
    bb.side = bb.max - bb.min ; 

    return gtransform ? bb.transform(gtransform->t) : bb ; 
}

npart ncylinder::part()
{
    npart p = nnode::part();
    assert( p.getTypeCode() == CSG_CYLINDER );
    return p ; 
}



