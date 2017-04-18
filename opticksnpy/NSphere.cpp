
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>


#include "NGLMExt.hpp"

// sysrap-
#include "OpticksCSG.h"

// npy-
#include "NSphere.hpp"
#include "NBBox.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"

#include "PLOG.hh"


float nsphere::costheta(float z)
{
   return (z - center.z)/radius ;  
}

// signed distance function

float nsphere::operator()(float x, float y, float z) 
{
    glm::vec4 p(x,y,z,1.f); 
    if(gtransform) p = gtransform->v * p ;  // v:inverse-transform
    return glm::distance( glm::vec3(p), center ) - radius ;
} 

glm::vec3 nsphere::gcenter()
{
    return gtransform == NULL ? center : glm::vec3( gtransform->t * glm::vec4(center, 1.f ) ) ; // t:transform
}


void nsphere::pdump(const char* msg, int verbosity)
{
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "no-label" )
              << " center " << center 
              << " radius " << radius 
              << " gcenter " << gcenter()
              << " gtransform " << !!gtransform 
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}



nbbox nsphere::bbox()
{
    nbbox bb = make_bbox();

    bb.min = make_nvec3(center.x - radius, center.y - radius, center.z - radius);
    bb.max = make_nvec3(center.x + radius, center.y + radius, center.z + radius);
    bb.side = bb.max - bb.min ; 

    return gtransform ? bb.transform(gtransform->t) : bb ; 
}


ndisc nsphere::intersect(nsphere& a, nsphere& b)
{
    // Find Z intersection disc of two Z offset spheres,
    // disc radius is set to zero when no intersection.
    //
    // http://mathworld.wolfram.com/Circle-CircleIntersection.html
    //
    // cf pmt-/ddpart.py 

    float R = a.radius ;
    float r = b.radius ; 

    // operate in frame of Sphere a 

    float dx = b.center.x - a.center.x ; 
    float dy = b.center.y - a.center.y ; 
    float dz = b.center.z - a.center.z ; 

    assert(dx == 0 && dy == 0 && dz != 0);

    float d = dz ;  
    float dd_m_rr_p_RR = d*d - r*r + R*R  ; 
    float z = dd_m_rr_p_RR/(2.f*d) ;
    float yy = (4.f*d*d*R*R - dd_m_rr_p_RR*dd_m_rr_p_RR)/(4.f*d*d)  ;
    float y = yy > 0 ? sqrt(yy) : 0 ;   


    nplane plane = make_plane(0,0,1,z + a.center.z) ;
    ndisc  disc = make_disc(plane, y) ;

    return disc ;      // return to original frame
}


npart nsphere::part()
{
    npart p = nnode::part();

    assert( p.getTypeCode() == CSG_SPHERE );

    if(npart::VERSION == 0u)
    {
        // TODO: move this belongs into a zsphere not here ???
        LOG(warning) << "nsphere::part override bbox " ;  
        float z = center.z ;  
        float r  = radius ; 
        nbbox bb = make_bbox(z - r, z + r, r, r);

        p.setBBox(bb);
    }
    return p ; 
}



npart nsphere::zlhs(const ndisc& dsk)
{
    npart p = part();

    float z = center.z ;  
    float r  = radius ; 
    nbbox bb = make_bbox(z - r, dsk.z(), -dsk.radius, dsk.radius);
    p.setBBox(bb);

    return p ; 
}

npart nsphere::zrhs(const ndisc& dsk)
{
    npart p = part();

    float z = center.z ;  
    float r  = radius ; 
    nbbox bb = make_bbox(dsk.z(), z + r, -dsk.radius, dsk.radius);
    p.setBBox(bb);

    return p ; 
}


