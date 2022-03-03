/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>

#include "NGLMExt.hpp"
#include "nmat4triple.hpp"

// sysrap-
#include "OpticksCSG.h"

// npy-
#include "NZSphere.hpp"
#include "NBBox.hpp"
#include "Nuv.hpp"

#include "PLOG.hh"



nzsphere* nzsphere::Create(const nquad& param, const nquad& param1 )
{
    nzsphere* n = new nzsphere ; 
    nnode::Init(n,CSG_ZSPHERE) ; 

    n->param = param ; 
    n->param1 = param1 ; 
    n->check();

    return n ; 
}

nzsphere* nzsphere::Create(float x, float y, float z, float radius, float z1, float z2 )
{
    nquad p0, p1 ; 
    p0.f = {x,y,z,radius} ;
    p1.f = {z1, z2, 0,0} ;
    return Create(p0, p1);
}

nzsphere* nzsphere::Create()
{
    return Create(0.f, 0.f, 0.f, 100.f, -50.f, 70.f );
}

nzsphere* nzsphere::Create(float x0, float y0, float z0, float w0, float x1, float y1, float z1, float w1 )
{
    // used by code generation 
    assert( z1 == 0.f );
    assert( w1 == 0.f );
    return Create( x0,y0,z0,w0,x1,y1 );
}





glm::vec3 nzsphere::center() const { return glm::vec3(x(),y(),z()) ;  } 

float nzsphere::x() const {      return param.f.x ; }
float nzsphere::y() const {      return param.f.y ; }
float nzsphere::z() const {      return param.f.z ; }
float nzsphere::radius() const { return param.f.w ; }

float nzsphere::z2() const {      return param1.f.y ; }  // z2 > z1
float nzsphere::z1() const {      return param1.f.x ; }
float nzsphere::r1() const {      return rz(z1()) ; } 
float nzsphere::r2() const {      return rz(z2()) ; } 

float nzsphere::zmax() const {    return z() + z2() ; }
float nzsphere::zmin() const {    return z() + z1() ; }
float nzsphere::zc() const {      return (zmin() + zmax())/2.f ; }

unsigned nzsphere::flags() const { return param2.u.x ; }

// grow the zsphere upwards on upper side (z2) or downwards on down side (z1)
void  nzsphere::increase_z2(float dz){ assert( dz >= 0.f) ; param1.f.y += dz ; check() ; } // z2 > z1
void  nzsphere::decrease_z1(float dz){ assert( dz >= 0.f) ; param1.f.x -= dz ; check() ; }
void  nzsphere::set_zcut(float z1, float z2){  param1.f.x = z1 ; param1.f.y = z2 ; check() ; }


float nzsphere::rz(float z_) const 
{
    float r = radius(); 
    return sqrt(r*r - z_*z_) ;  
}

/*
In [23]: np.arccos([1,0,-1])/np.pi
Out[23]: array([ 0. ,  0.5,  1. ])
*/ 

float nzsphere::startTheta() const { return acosf(z2()/radius()); }
float nzsphere::endTheta() const {   return acosf(z1()/radius()); }
float nzsphere::deltaTheta() const { return endTheta() - startTheta(); }






void nzsphere::check() const 
{
    bool z2_gt_z1 = z2() > z1() ; 
    bool z2_lt_radius = fabs(z2()) <= radius() ; 
    bool z1_lt_radius = fabs(z1()) <= radius() ; 
    bool zmax_gt_zmin = zmax() > zmin() ; 

    if(!z2_gt_z1)
       LOG(fatal) 
          << " NOT z2_gt_z1 " 
          << " z1 " << z1()
          << " z2 " << z2()
          ;
 
    if(!z2_lt_radius)
       LOG(fatal) 
           << " NOT z2_lt_radius "
           << " z2 " << z2() 
           << " radius " << radius() 
           ;

    if(!z1_lt_radius)
       LOG(fatal) 
           << " NOT z1_lt_radius "
           << " z1 " << z1() 
           << " radius " << radius() 
           ;

    if(!zmax_gt_zmin)
       LOG(fatal) 
           << " NOT zmax_gt_zmin "
           << " zmax " << zmax() 
           << " zmin " << zmin() 
           ;

    assert( z2_gt_z1 );

    if(!z2_lt_radius) LOG(fatal) << " tmp skip assert "; 
    //assert( z2_lt_radius );

    assert( z1_lt_radius );
    assert( zmax_gt_zmin ); 
}





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



