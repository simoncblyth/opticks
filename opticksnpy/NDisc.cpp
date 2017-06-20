
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>


#include "NGLMExt.hpp"

// sysrap-
#include "OpticksCSG.h"

// npy-
#include "NDisc.hpp"
#include "NBBox.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"

#include "PLOG.hh"

    
nbbox ndisc::bbox() const 
{
    nbbox bb = make_bbox();

    bb.max = make_nvec3(center.x + radius, center.y + radius, z2 );
    bb.min = make_nvec3(center.x - radius, center.y - radius, z1 );
    bb.side = bb.max - bb.min ; 
    bb.invert = complement ; 
    bb.empty = false ; 

    return gtransform ? bb.transform(gtransform->t) : bb ; 
}


float ndisc::operator()(float x, float y, float z) const 
{
    glm::vec4 p(x,y,z,1.0); 
    if(gtransform) p = gtransform->v * p ; 

    float dinf = glm::distance( glm::vec2(p.x, p.y), glm::vec2(center.x, center.y) ) - radius ;  // <- no z-dep

    float qcap_z = z2 ; 
    float pcap_z = z1 ; 

    float d_PQCAP = fmaxf( p.z - qcap_z, -(p.z - pcap_z) );

    float sd = fmaxf( d_PQCAP, dinf );
/*
    std::cout 
          << "ndisc" 
          << " p " << p 
          << " dinf " << dinf
          << " dcap " << dcap
          << " sd " << sd
          << std::endl 
          ;
*/
    return complement ? -sd : sd ; 
} 



glm::vec3 ndisc::gseedcenter() const 
{
    return gtransform == NULL ? center : glm::vec3( gtransform->t * glm::vec4(center, 1.f ) ) ;
}

glm::vec3 ndisc::gseeddir()
{
    glm::vec4 dir(1,0,0,0);   // Z: not a good choice as without endcap fail to hit 
    if(gtransform) dir = gtransform->t * dir ; 
    return glm::vec3(dir) ;
}


void ndisc::pdump(const char* msg ) const 
{
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "no-label" )
              << " center " << center 
              << " radius " << radius 
              << " z1 " << z1
              << " z2 " << z2
              << " gseedcenter " << gseedcenter()
              << " gtransform " << !!gtransform 
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}

npart ndisc::part()
{
    npart p = nnode::part();
    assert( p.getTypeCode() == CSG_DISC );
    return p ; 
}


