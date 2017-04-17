
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


float ncylinder::operator()(float x, float y, float z) 
{
    glm::vec4 p(x,y,z,1.0); 
    if(gtransform) p = gtransform->v * p ; 

    float dinf = glm::distance( glm::vec2(p.x, p.y), glm::vec2(center.x, center.y) ) - radius ;  // <- no z-dep
    float dcap = fabs(p.z) - sizeZ/2.f ;  
    // hmm this doesnt honour the endcap flags ... it 

    float sd = fmaxf( dinf, dcap ) ;  

/*
    std::cout 
          << "ncylinder" 
          << " p " << p 
          << " dinf " << dinf
          << " dcap " << dcap
          << " sd " << sd
          << std::endl 
          ;
*/

    return sd ; 
} 


/*

See env-;sdf-

http://iquilezles.org/www/articles/distfunctions/distfunctions.htm

float sdCappedCylinder( vec3 p, vec2 h )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - h;
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

http://mercury.sexy/hg_sdf/



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
              << " radius " << radius 
              << " sizeZ " << sizeZ
              << " flags " << flags
              << " gcenter " << gcenter()
              << " gtransform " << !!gtransform 
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}


nbbox ncylinder::bbox()
{
    nbbox bb = make_nbbox();

    bb.min = make_nvec3(center.x - param.f.w, center.y - param.f.w, center.z - param.f.w);
    bb.max = make_nvec3(center.x + param.f.w, center.y + param.f.w, center.z + param.f.w);
    bb.side = bb.max - bb.min ; 

    return gtransform ? bb.transform(gtransform->t) : bb ; 
}

npart ncylinder::part()
{
    npart p = nnode::part();
    assert( p.getTypeCode() == CSG_CYLINDER );
    return p ; 
}



