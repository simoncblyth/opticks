
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>


#include "NGLMExt.hpp"

// sysrap-
#include "OpticksCSG.h"

// npy-
#include "NTorus.hpp"
#include "NBBox.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"
#include "Nuv.hpp"

#include "PLOG.hh"


// signed distance function

float ntorus::operator()(float x_, float y_, float z_) const 
{
    glm::vec4 p(x_,y_,z_,1.f); 
    if(gtransform) p = gtransform->v * p ;  // v:inverse-transform
    
    glm::vec2 q( glm::length(glm::vec2(p)) - rmajor() , p.z );
    float sd = glm::length(q) - rminor() ;

    return complement ? -sd : sd ;
} 



glm::vec3 ntorus::gseeddir()
{
    glm::vec4 dir(1,0,0,0); 
    if(gtransform) dir = gtransform->t * dir ; 
    return glm::vec3(dir) ;
}


unsigned ntorus::par_nsurf() const 
{
   return 1 ; 
}
int ntorus::par_euler() const 
{
   return 2 ; 
}
unsigned ntorus::par_nvertices(unsigned nu, unsigned nv) const 
{
   // expected unique vertex count, accounting for extra ones, poles and 360-seam 
   assert( nv > 2 ); 
   return 2 + (nu+1-1)*(nv+1-2) ;     
}


glm::vec3 ntorus::par_pos_model(const nuv& uv) const 
{
    unsigned s  = uv.s(); 
    assert(s == 0);

    glm::vec3 c = center();
    glm::vec3 pos(c);

    float R_ = rmajor();
    float r_ = rminor();

    _par_pos_body( pos, uv, R_, r_ );
  
    return pos ; 
}


void ntorus::_par_pos_body(glm::vec3& pos,  const nuv& uv, const float R_, const float r_ )  // static
{
    bool seamed = true ; 
    float u_ = uv.fu2pi(seamed); 
    float v_ = uv.fv2pi(seamed) ; 

    float cu = cosf(u_);
    float su = sinf(u_);

    float cv = cosf(v_);
    float sv = sinf(v_);

    // http://mathworld.wolfram.com/Torus.html

    pos += glm::vec3((R_+r_*cv)*cu, (R_+r_*cv)*su, r_*sv );
  
}



glm::vec3 ntorus::gseedcenter() const 
{
    glm::vec3 c = center();
    return gtransform == NULL ? c : glm::vec3( gtransform->t * glm::vec4(c, 1.f ) ) ; // t:transform
}


void ntorus::pdump(const char* msg) const 
{
    nnode::dump();
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "no-label" )
              << " center " << center()
              << " rmajor " << rmajor()
              << " rminor " << rminor()
              << " gseedcenter " << gseedcenter()
              << " gtransform " << !!gtransform 
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}



nbbox ntorus::bbox() const 
{
    const float r_ = rminor() ;
    const float R_ = rmajor() ;

    float  rsum = r_ + R_  ; 

    glm::vec3 mi( -rsum,  -rsum,  -r_ );
    glm::vec3 mx(  rsum,   rsum,   r_ );
    nbbox bb = make_bbox(mi, mx, complement);

    return gtransform ? bb.make_transformed(gtransform->t) : bb ; 
}


