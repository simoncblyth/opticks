
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

#include "NCylinder.h"


nbbox ncylinder::bbox() const 
{
    nbbox bb = make_bbox();

    float r = radius();
    glm::vec3 c = center();

    bb.max = make_nvec3(c.x + r, c.y + r, z2() );
    bb.min = make_nvec3(c.x - r, c.y - r, z1() );

    bb.side = bb.max - bb.min ; 
    bb.invert = complement ; 
    bb.empty = false ; 

    return gtransform ? bb.transform(gtransform->t) : bb ; 
}

float ncylinder::operator()(float x_, float y_, float z_) const 
{
    glm::vec4 p(x_,y_,z_,1.0); 
    if(gtransform) p = gtransform->v * p ; 

    float dinf = glm::distance( glm::vec2(p.x, p.y), glm::vec2(x(), y()) ) - radius() ;  // <- no z-dep
    float qcap_z = z2() ; 
    float pcap_z = z1() ; 

    float d_PQCAP = fmaxf( p.z - qcap_z, -(p.z - pcap_z) );

    float sd = fmaxf( d_PQCAP, dinf );

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
    return complement ? -sd : sd ; 
} 

glm::vec3 ncylinder::gseedcenter() const 
{
    return gtransform == NULL ? center() : glm::vec3( gtransform->t * glm::vec4(center(), 1.f ) ) ;
}

glm::vec3 ncylinder::gseeddir() const 
{
    glm::vec4 dir(1,0,0,0);   // Z: not a good choice as without endcap fail to hit 
    if(gtransform) dir = gtransform->t * dir ; 
    return glm::vec3(dir) ;
}

void ncylinder::pdump(const char* msg ) const 
{
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "no-label" )
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
npart ncylinder::part() const 
{
    npart p = nnode::part();
    assert( p.getTypeCode() == CSG_CYLINDER );
    return p ; 
}
*/






/*

Can SDFs model finite open cylinder, ie no endcaps or 1 endcap  ?
====================================================================

* i do not think so...  but then CSG cannot do this either

* suspect this is fundamental limitation of geometry modelling with SDF,
  ... **can only handle closed geometry** 

  * yep that is definitiely the case for CSG  *S* means SOLID,


Extract from env-;sdf-:

Slab is a difference of half-spaces

* sdfA = z - h      (plane at z = h) 
* sdfB = z + h      (plane at z = -h ),  
* ~sdfB = -(z+h)    (same position, but now inside are upwards to +z)

::

    intersect(sdfA, ~sdfB) 
    max( z - h , -(z + h) )
    max( z - h , -z - h )
    max(z, -z) - h
    abs(z) - h 


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


