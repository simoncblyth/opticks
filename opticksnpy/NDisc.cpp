
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
#include "NCone.hpp"
#include "Nuv.hpp"

#include "PLOG.hh"

    
nbbox ndisc::bbox() const 
{

    float r = radius() ;
    glm::vec3 c = center(); 

    glm::vec3 mx(c.x + r, c.y + r, z2() );
    glm::vec3 mi(c.x - r, c.y - r, z1() );

    nbbox bb = make_bbox(mi, mx, complement);

    return gtransform ? bb.make_transformed(gtransform->t) : bb ; 
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






unsigned ndisc::par_nsurf() const 
{
   return 3 ; 
}
int ndisc::par_euler() const 
{
   return 2 ; 
}
unsigned ndisc::par_nvertices(unsigned /*nu*/, unsigned /*nv*/) const 
{
   return 0 ; 
}

glm::vec3 ndisc::par_pos_model(const nuv& uv) const 
{
    // same as ncylinder

    unsigned s  = uv.s(); 
    assert(s < par_nsurf());

    float r1_ = radius();
    float r2_ = radius();
    float z1_ = z1();
    float z2_ = z2();

    assert( z2_ > z1_ );

    glm::vec3 pos(0,0,0);
    pos.x = x();
    pos.y = y();
    // start on axis

    switch(s)
    {
       case 0:  ncone::_par_pos_body(  pos, uv, r1_ ,  z1_ , r2_ , z2_ ) ; break ; 
       case 1:  nnode::_par_pos_endcap(pos, uv, r2_ ,  z2_ )             ; break ; 
       case 2:  nnode::_par_pos_endcap(pos, uv, r1_ ,  z1_ )             ; break ; 
    }
    return pos ; 
}








/*
npart ndisc::part() const 
{
    npart p = nnode::part();
    assert( p.getTypeCode() == CSG_DISC );
    return p ; 
}
*/

