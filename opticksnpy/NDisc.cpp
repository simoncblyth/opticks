
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

    float r = radius() ;
    glm::vec3 c = center(); 

    bb.max = make_nvec3(c.x + r, c.y + r, z2() );
    bb.min = make_nvec3(c.x - r, c.y - r, z1() );
    bb.side = bb.max - bb.min ; 
    bb.invert = complement ; 
    bb.empty = false ; 

    return gtransform ? bb.transform(gtransform->t) : bb ; 
}


float ndisc::operator()(float x_, float y_, float z_) const 
{
    glm::vec4 p(x_,y_,z_,1.0); 
    if(gtransform) p = gtransform->v * p ; 

    float r = radius();  
    glm::vec3 c = center(); 

    float dinf = glm::distance( glm::vec2(p.x, p.y), glm::vec2(c.x, c.y) ) - r ;  // <- no z-dep

    float qcap_z = z2() ; 
    float pcap_z = z1() ; 

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
    return gtransform == NULL ? center() : glm::vec3( gtransform->t * glm::vec4(center(), 1.f ) ) ;
}

glm::vec3 ndisc::gseeddir() const 
{
    glm::vec4 dir(1,0,0,0);   // Z: not a good choice as without endcap fail to hit 
    if(gtransform) dir = gtransform->t * dir ; 
    return glm::vec3(dir) ;
}


void ndisc::pdump(const char* msg ) const 
{
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "-" )
              << " center " << center() 
              << " radius " << radius() 
              << " z1 " << z1()
              << " z2 " << z2()
              << " gseedcenter " << gseedcenter()
              << " gtransform " << !!gtransform 
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}


/*
npart ndisc::part() const 
{
    npart p = nnode::part();
    assert( p.getTypeCode() == CSG_DISC );
    return p ; 
}
*/

