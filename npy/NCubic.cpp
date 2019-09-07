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

// sysrap-
#include "OpticksCSG.h"

// npy-
#include "NCubic.hpp"
#include "NBBox.hpp"

#include "NPart.hpp"
#include "Nuv.hpp"

#include "PLOG.hh"

/*

NCubic Surface of Revolution
=====================================

* x^2 + y^2  = rr =  A z^3 + B z^2 + C z + D   

*  d(rr)/dz = 3 A z^2 + 2 B z + C 

Local minimum or maximum at 2 z values, from quadratic roots of derivative

   -B +- sqrt( B^2 - 3 A C )
  --------------------------
        3 A


Contrast with hyperboloid and torus:
                    
* x^2 +  y^2  =  r0^2 * (  (z/zf)^2  +  1 )

* x^2 + y^2  =  ( R - sqrt( r^2 - z^2 ) )^2

*/
// signed distance function
float ncubic::operator()(float x_, float y_, float z_) const 
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


unsigned ncubic::par_nsurf() const 
{
   return 1 ; 
}
int ncubic::par_euler() const 
{
   return 2 ; 
}
unsigned ncubic::par_nvertices(unsigned nu, unsigned nv) const 
{
   // expected unique vertex count, accounting for extra ones, poles and 360-seam 
   assert( nv > 2 ); 
   return 2 + (nu+1-1)*(nv+1-2) ;     
}


glm::vec3 ncubic::par_pos_model(const nuv& uv) const 
{
    unsigned s  = uv.s(); 
    assert(s == 0);
    glm::vec3 pos(0.f);
    return pos ; 
}


void ncubic::pdump(const char* msg) const 
{
    nnode::dump();
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "no-label" )
              << " center " << center()
              << " A " << A()
              << " B " << B()
              << " C " << C()
              << " D " << D()
              << " z1 " << z1()
              << " z2 " << z2()
              << " rrz(z1) " << rrz(z1())
              << " rrz(z2) " << rrz(z2())
              << " rrmax() " << rrmax()
              << " gseedcenter " << gseedcenter()
              << " gtransform " << !!gtransform 
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}

float ncubic::rrmax() const 
{
/*
Maximum radius^2 within z range, either from 
local max "bulge" that is the domain max 
or z1/z2 endpoint max.

* x^2 + y^2  = rr =  A z^3 + B z^2 + C z + D   

*  d(rr)/dz = 3 A z^2 + 2 B z + C 

Local minimum or maximum at 2 z values, from quadratic roots of derivative

   -B +- sqrt( B^2 - 3 A C )
  --------------------------
        3 A
*/

    float z1_ = z1() ; 
    float z2_ = z2() ; 

    float d = 3.f*A() ; 
    float b = B() ; 
    float c = C() ; 

    float disc = b*b - d*c ; 
    float sdisc = disc > 0.f ? sqrt(disc) : 0.f ; 

    float q = b > 0.f ? -(b + sdisc) : -(b - sdisc) ;
    float e1 = q/d  ;
    float e2 = c/q  ;

    float rr[4] ; 
    rr[0] = rrz(z1_) ; 
    rr[1] = rrz(z2_) ;
    rr[2] = disc > 0.f && e1 > z1_ && e1 < z2_ ? rrz(e1) : 0.f ; 
    rr[3] = disc > 0.f && e2 > z1_ && e2 < z2_ ? rrz(e2) : 0.f ; 
  
    float rrmax_ = 0.f ;   
    for(unsigned i=0 ; i < 4 ; i++) if(rr[i] > rrmax_) rrmax_ = rr[i] ; 

    return rrmax_ ; 
}

nbbox ncubic::bbox() const 
{
    float z1_ = z1() ; 
    float z2_ = z2() ; 
    float rmx = sqrt(rrmax()) ; 

    glm::vec3 mi( -rmx,  -rmx, z1_ );
    glm::vec3 mx(  rmx,   rmx, z2_ );
    nbbox bb = make_bbox(mi, mx, complement);

    return gtransform ? bb.make_transformed(gtransform->t) : bb ; 
}


