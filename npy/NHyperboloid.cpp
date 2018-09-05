
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>


#include "NGLMExt.hpp"

// sysrap-
#include "OpticksCSG.h"

// npy-
#include "NHyperboloid.hpp"
#include "NBBox.hpp"

#include "NPart.hpp"
#include "Nuv.hpp"

#include "PLOG.hh"

/*

One sheet, hyperbolic hyperboloid
=====================================

    x^2     y^2      z^2
    --- +  ----  -  ----   =   1      
    a^2     b^2      c^2 

Specialize to surface of revolution,  a = b 
    
    x^2     y^2      z^2
    --- +  ----  -  ----   =   1      
    a^2     a^2      c^2 


    x^2 +  y^2  =  a^2 * (  (z/zf)^2  +  1 )
            
    x^2 +  y^2  =  a^2            # at z = 0     (so param r0, waist radius is "a" ) 
 
    x^2 +  y^2  =  a^2  * 2       # at z = zf   radius increases by sqrt(2)      
                     


    x^2 +  y^2  =  r0^2 * (  (z/zf)^2  +  1 )



Getting close to a torus(r,R) neck shape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    r0 = R - r   (at z=0)    waist radius of hyperboloid
    w  = R       (at z=r)
    
Consider targetting a particular x^2 + y^2 =  ww 
What zf is required to hit radius^2 of ww at z = zw     


     ww = r0^2 ( (zw/zf)^2 + 1 )

      ww - rr0
     ------------- =  (zw/zf)^2  
         rr0
 
                          rr0
        zf = zw * sqrt( -----------   )
                        ww - rr0
    

 torus
~~~~~~~~~~

    (R - sqrt(x^2 + y^2))^2 = r^2 - z^2  

     R - sqrt( x^2 + y^2)  =  sqrt( r^2 - z^2 )

   
     sqrt(x^2 + y^2 ) =  R - sqrt( r^2 - z^2 )

     sqrt(x^2 + y^2 ) =  R -+ r     at z=0

     sqrt(x^2 + y^2 ) =  R          at z=r 


 




*/
// signed distance function


float nhyperboloid::operator()(float x_, float y_, float z_) const 
{
    glm::vec4 p(x_,y_,z_,1.f); 
    if(gtransform) p = gtransform->v * p ;  // v:inverse-transform

    // estimate distance using cylinder appropriate for p.z of query point
    // and endcaps

    float r = rz(p.z) ; 
    float dinf = glm::distance( glm::vec2(p.x, p.y), glm::vec2(0, 0) ) - r ;  // <- no z-dep

    float qcap_z = z2() ;  // typically +ve   z2>z1  
    float pcap_z = z1() ;  // typically -ve
    float d_PQCAP = fmaxf( p.z - qcap_z, -(p.z - pcap_z) );

    float sd = fmaxf( d_PQCAP, dinf );

    return complement ? -sd : sd ;
} 

float nhyperboloid::rz(float z) const
{
    return sqrt(rrz(z)) ; 
}

float nhyperboloid::rrz(float z) const
{ 
    // r^2 at z 
    //  rr =  x^2 +  y^2  =  a^2 * (  (z/c)^2  +  1 )
    float a = r0();
    float aa = a*a ; 
    float zs = z/zf() ; 
    return aa * ( zs*zs + 1.f ) ;  
} 

unsigned nhyperboloid::par_nsurf() const 
{
   return 1 ; 
}
int nhyperboloid::par_euler() const 
{
   return 2 ; 
}
unsigned nhyperboloid::par_nvertices(unsigned nu, unsigned nv) const 
{
   // expected unique vertex count, accounting for extra ones, poles and 360-seam 
   assert( nv > 2 ); 
   return 2 + (nu+1-1)*(nv+1-2) ;     
}


glm::vec3 nhyperboloid::par_pos_model(const nuv& uv) const 
{
    unsigned s  = uv.s(); 
    assert(s == 0);
    glm::vec3 pos(0.f);
    return pos ; 
}

glm::vec3 nhyperboloid::gseedcenter() const 
{
    glm::vec3 c = center();
    return gtransform == NULL ? c : glm::vec3( gtransform->t * glm::vec4(c, 1.f ) ) ; // t:transform
}


void nhyperboloid::pdump(const char* msg) const 
{
    nnode::dump();
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "no-label" )
              << " center " << center()
              << " r0 " << r0()
              << " rrz(0) " << rrz(0)
              << " rrz(z1) " << rrz(z1())
              << " rrz(z2) " << rrz(z2())
              << " gseedcenter " << gseedcenter()
              << " gtransform " << !!gtransform 
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}

nbbox nhyperboloid::bbox() const 
{
    float z1_ = z1() ; 
    float z2_ = z2() ; 

    float rr1=rrz(z1_) ;
    float rr2=rrz(z2_) ;
    float rmx = sqrt(fmaxf( rr1, rr2 )) ; 

    glm::vec3 mi( -rmx,  -rmx, z1_ );
    glm::vec3 mx(  rmx,   rmx, z2_ );
    nbbox bb = make_bbox(mi, mx, complement);

    return gtransform ? bb.make_transformed(gtransform->t) : bb ; 
}


