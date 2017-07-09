
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>

#include "NGLMExt.hpp"

// sysrap-
#include "OpticksCSG.h"

// npy-
#include "NZSphere.hpp"
#include "NBBox.hpp"
#include "Nuv.hpp"

#include "PLOG.hh"


float nzsphere::operator()(float x_, float y_, float z_) const 
{
    glm::vec4 p(x_,y_,z_,1.f); 
    if(gtransform) p = gtransform->v * p ;  // v:inverse-transform

    glm::vec3 c = center(); 
    float r = radius();

    float d_sphere = glm::distance( glm::vec3(p), c ) - r ;

    float d_slab = fmaxf( p.z - zmax(), -(p.z - zmin()) );  

    float sd = fmaxf( d_sphere, d_slab );  // CSG intersect of sphere with slab

    return complement ? -sd : sd ; 
} 

nbbox nzsphere::bbox() const 
{
    glm::vec3 c = center(); 
    float r = radius(); 

    glm::vec3 mx(c.x + r, c.y + r, zmax() );
    glm::vec3 mi(c.x - r, c.y - r, zmin() );

    nbbox bb = make_bbox(mi, mx, complement);

    return gtransform ? bb.make_transformed(gtransform->t) : bb ; 
}

glm::vec3 nzsphere::gseedcenter() const 
{
    glm::vec3 c = center(); 
    glm::vec4 seedcenter( c.x, c.y, zc(), 1.f ); 
    return apply_gtransform(seedcenter);
}

glm::vec3 nzsphere::gseeddir() const 
{
    glm::vec4 dir(0,0,1,0); 
    return apply_gtransform(dir);
}



void nzsphere::pdump(const char* msg) const 
{
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "-" )
              << " center " << center()
              << " radius " << radius()
              << " zmin " << zmin()
              << " zmax " << zmax()
              << " z1 " << z1()
              << " z2 " << z2()
              << " zc " << zc()
              << " gseedcenter " << gseedcenter()
              << " gtransform " << !!gtransform 
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}







unsigned nzsphere::par_nsurf() const 
{
    unsigned n = 1 ;  
    if(has_z2_endcap()) n++ ; 
    if(has_z1_endcap()) n++ ; 
    return n ; 
}

bool nzsphere::has_z2_endcap() const 
{
    float r_ = radius();
    float z2_ = z2(); 
    return fabsf(z2_) < r_ ; 
}

bool nzsphere::has_z1_endcap() const 
{
    float r_ = radius();
    float z1_ = z1(); 
    return fabsf(z1_) < r_ ; 
}


int nzsphere::par_euler() const 
{
   return 2 ; 
}
unsigned nzsphere::par_nvertices(unsigned /*nu*/, unsigned /*nv*/) const 
{
   return 0 ;     
}

glm::vec3 nzsphere::par_pos_model(const nuv& uv) const 
{
    unsigned s  = uv.s(); 
    assert(s == 0 || s == 1 || s == 2);

    float z1_ = z1(); 
    float z2_ = z2(); 

    float r_ = radius();
    float rz1_ = rz(z1_);
    float rz2_ = rz(z2_);

    glm::vec3 pos(0,0,0);
    pos.x = x();
    pos.y = y();

    // start on axis

    unsigned ncap = par_nsurf() - 1 ; 

    bool has_z1cap_ = has_z1_endcap() ;
    bool has_z2cap_ = has_z2_endcap() ;


   float kludge = 0.1f ;   // avoid NOpenMesh complex edge problem by leaving a gap

    if( ncap == 2 )
    {
        assert( s == 0 || s== 1 || s == 2 );
        switch(s)
        {
           case 0: nzsphere::_par_pos_body(  pos, uv, r_ ,  z1_ , z2_ - kludge, has_z1cap_, has_z2cap_ ) ; break ;
           case 1: nnode::_par_pos_endcap(   pos, uv, rz2_ ,  z2_ )       ; break ;
           case 2: nnode::_par_pos_endcap(   pos, uv, rz1_ ,  z1_ )       ; break ;
        } 
    }
    else if( ncap == 1)
    {
        assert( s == 0 || s== 1 );

        if(s == 0)
        {
            nzsphere::_par_pos_body(  pos, uv, r_ ,  z1_ , z2_, has_z1cap_, has_z2cap_ ) ;
        }
        else
        {
            assert( has_z1cap_ ^ has_z2cap_ );
            float zcap = has_z1cap_ ? z1_ : z2_ ; 
            float rzcap = has_z1cap_ ? rz1_ : rz2_ ;  

            nnode::_par_pos_endcap(pos, uv, rzcap ,  zcap ) ;
        }
    } 
    else if( ncap == 0)
    {
        assert( s == 0 );
        nzsphere::_par_pos_body(  pos, uv, r_ ,  z1_ , z2_, has_z1cap_, has_z2cap_ ) ;
    }
    return pos ; 
}



void nzsphere::_par_pos_body(glm::vec3& pos,  const nuv& uv, const float r_, const float z1_ , const float z2_, const bool has_z1cap_, const bool has_z2cap_ )  // static
{
    unsigned  v  = uv.v(); 
    unsigned nv  = uv.nv(); 

    bool is_z2v = v == 0 ; 
    bool is_z1v = v == nv ; 

    if( !has_z2cap_ && is_z2v )
    {
        pos += glm::vec3(0,0,z2_ ) ; 
    }
    else if( !has_z1cap_ && is_z1v )
    {
        pos += glm::vec3(0,0,z1_ ) ; 
    }
    else
    {
        bool seamed = true ; 
        float azimuth = uv.fu2pi(seamed); 

        float cp_z1 = z1_/r_ ; 
        float cp_z2 = z2_/r_ ; 
        float cp_delta = cp_z2 - cp_z1 ; 
        float cp = cp_z1 + uv.fv()*cp_delta ;   // linear steps between cos(polar) extremes ?

        float polar = acosf(cp);
        float sp = sinf(polar) ;
   
        float ca = cosf(azimuth);
        float sa = sinf(azimuth);

        pos += glm::vec3( r_*ca*sp, r_*sa*sp, r_*cp );
    }
}

/*
full sphere 

        fv 0 -> 1  
        fvpi 0 -> pi
        cos(fvpi) 1 -> -1

zsphere  
         z1 -> z2      z2>z1   |z1| <= r, |z2| <= r  
         c2 = z2/r   c1=z1/r


*/




