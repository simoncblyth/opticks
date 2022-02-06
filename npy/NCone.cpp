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
#include "NCone.hpp"
#include "NBBox.hpp"
#include "NPart.hpp"
#include "Nuv.hpp"

#include "PLOG.hh"


float ncone::r1() const { return param.f.x ; }
float ncone::z1() const { return param.f.y ; }
float ncone::r2() const { return param.f.z ; }
float ncone::z2() const { return param.f.w ; }  // z2 > z1

// grow the cone on upwards on upper side (z2) or downwards on down side (z1)

/**
ncone::decrease_z1 ncone::increase_z2
-----------------------------------------

TODO: avoid increase_z1 and decrease_z2 changing angle of the cone by proportionately changing r1 and r2 


                    
                   new_r2
                    |
                    | r2   
                    | | 
                    | |
              +~~~~~+ |        new_z2
             /   .   \|
            +----.----+        z2      z2 > z1  by assertion 
           /     .     \
          /      .      \
         /       .       \
        +--------.--------+    z1
       /         .        |\
      +~~~~~~~~~~.~~~~~~~~|~+  new_z1 
                          | |
                          | |
                          r1| 
                            |
                           new_r1


Consider increasing z2 by dz (dz > 0) to new_z2
while keeping the same cone angle

       new_z2 - z1       z2 - z1  
      -------------  =  ----------
       new_r2 - r1       r2 - r1 


      new_r2 - r1       z2i - z1  
      ----------  =  ----------
       r2  - r1        z2 - z1 

       new_r2 =   r1 +  (r2 - r1) ( new_z2 - z1 )/(z2 - z1)


similarly for decreasing z1 by dz (dz > 0) to new_z1


     new_r1 - r2          r2 - r1
   ---------------  =   ----------
     new_z1 - z2          z2 - z1 


        
     new_r1 = r2 + ( r2 - r1 )*(new_z1 - z2) / (z2 - z1 )


**/


void  ncone::increase_z2(float dz)
{ 
    assert( dz >= 0.f) ; 

    const float _r1 = param.f.x ; 
    const float _z1 = param.f.y ; 
    const float _r2 = param.f.z ; 
    const float _z2 = param.f.w ; 

    const float new_z2 = _z2 + dz ; 
    const float new_r2 =  _r1 +  (_r2 - _r1)*( new_z2 - _z1 )/(_z2 - _z1)  ; 

    LOG(info) 
        << " new_z2 " << new_z2 
        << " new_r2 " << new_r2 
        ;


    param.f.z = new_r2 < 0.f ? 0.f : new_r2 ; 
    param.f.w = new_z2 ; 
} 


void  ncone::decrease_z1(float dz)
{ 
    assert( dz >= 0.f) ; 

    const float _r1 = param.f.x ; 
    const float _z1 = param.f.y ; 
    const float _r2 = param.f.z ; 
    const float _z2 = param.f.w ; 

    const float new_z1 = _z1 - dz ; 
    const float new_r1 = _r2 + (_r2 - _r1)*(new_z1 - _z2) /(_z2 - _z1 ) ; 

    LOG(info) 
        << " new_z1 " << new_z1 
        << " new_r1 " << new_r1 
        ;


    param.f.x = new_r1 < 0.f ? 0.f : new_r1 ; 
    param.f.y = new_z1 ; 
}


float ncone::zc() const { return (z1() + z2())/2.f ; }
float ncone::rmax() const { return fmaxf( r1(), r2())  ; }

float ncone::z0() const {  return (z2()*r1()-z1()*r2())/(r1()-r2()) ; }


float ncone::tantheta() const { return (r2()-r1())/(z2()-z1()) ; }
float ncone::x() const { return 0.f ; }
float ncone::y() const { return 0.f ; }
glm::vec3 ncone::center() const { return glm::vec3(x(),y(),zc()) ; } 
glm::vec2 ncone::cnormal() const { return glm::normalize( glm::vec2(z2()-z1(),r1()-r2()) ) ; }
glm::vec2 ncone::csurface() const { glm::vec2 cn = cnormal() ; return glm::vec2( cn.y, -cn.x ) ; }      




/**
                    
                     
                 O         
                ...     
               . . .   
              .  .  . r2                
             .   .   .|
            +----.----+        z2      z2 > z1  by assertion 
           /     .     \
          /      .      \
         /       .       \
        +--------.--------+    z1
                 .        |   
                 .        r1 
                 .            
                             

                 r2 = 0 
                     
                 O      z2 = 0 
                /.\
               / . \
              /  .th\
             /   .   \
            /    .    \
           /     .     \
          /      .      \
         /       .       \
        +--------.--------+    z1          
                 .        |   
                 .        r1 
                 .            
 

               r2 - r1
    tan(th) = ---------- 
               z2 - z1 


        
       





**/




    


nbbox ncone::bbox() const 
{
    glm::vec3 mx(  rmax(),  rmax(), z2() );
    glm::vec3 mi( -rmax(), -rmax(), z1() );

    nbbox bb = make_bbox(mi, mx, complement);

    return gtransform ? bb.make_transformed(gtransform->t) : bb ; 
}

/**

* https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm


float sdCone( in vec3 p, in vec2 c, float h )
{
  // c is the sin/cos of the angle, h is height
  // Alternatively pass q instead of (c,h),
  // which is the point at the base in 2D
  vec2 q = h*vec2(c.x/c.y,-1.0);
    
  vec2 w = vec2( length(p.xz), p.y );
  vec2 a = w - q*clamp( dot(w,q)/dot(q,q), 0.0, 1.0 );
  vec2 b = w - q*vec2( clamp( w.x/q.x, 0.0, 1.0 ), 1.0 );
  float k = sign( q.y );
  float d = min(dot( a, a ),dot(b, b));
  float s = max( k*(w.x*q.y-w.y*q.x),k*(w.y-q.y)  );
  return sqrt(d)*sign(s);
}

**/

float ncone::operator()(float x_, float y_, float z_) const 
{
    glm::vec4 p(x_,y_,z_,1.0); 
    if(gtransform) p = gtransform->v * p ; 

    // Z: cone axis

    glm::vec2 q( glm::length(glm::vec2(p)), p.z-z0() ) ;        // cone coordinate (radial,axial) from apex origin  

    float sd_cone = glm::dot( cnormal(), q ) ; 

    float dz1 = p.z - z1()  ; 
    float dz2 = p.z - z2()  ;   // z2 > z1

    float sd_zslab = fmaxf( dz2, -dz1 );

    float sd = fmaxf( sd_cone, sd_zslab ); 

/*
    std::cout << "ncone::operator" 
              << " q (" << q.x << " " << q.y << ")"
              << " cnormal (" << cnormal.x << " " << cnormal.y << ")"
              << " sd " << sd
              << std::endl ; 
*/

    return complement ? -sd : sd  ; 
} 


glm::vec3 ncone::gseedcenter() const 
{
    return gtransform == NULL ? center() : glm::vec3( gtransform->t * glm::vec4(center(), 1.f ) ) ;
}

glm::vec3 ncone::gseeddir() const 
{
    glm::vec4 dir(1,0,0,0);  
    if(gtransform) dir = gtransform->t * dir ; 
    return glm::vec3(dir) ;
}


void ncone::pdump(const char* msg ) const 
{
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "-" )
              << " center " << center() 
              << " r1 " << r1()
              << " r2 " << r2()
              << " rmax " << rmax()
              << " z1 " << z1()
              << " z2 " << z2()
              << " zc " << zc()
              << " z0(apex) " << z0()
              << " gseedcenter " << gseedcenter()
              << " gtransform " << !!gtransform 
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}











unsigned ncone::par_nsurf() const 
{
   return 3 ; 
}
int ncone::par_euler() const 
{
   return 2 ; 
}
unsigned ncone::par_nvertices(unsigned /*nu*/, unsigned /*nv*/) const 
{
   return 0 ; 
}

glm::vec3 ncone::par_pos_model(const nuv& uv) const 
{
    unsigned s  = uv.s(); 
    assert(s < par_nsurf());

    float r1_ = r1();
    float r2_ = r2();
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




void ncone::_par_pos_body(glm::vec3& pos,  const nuv& uv, const float r1_, const float z1_,  const float r2_, const float z2_)  // static
{
    unsigned s  = uv.s(); 
    assert( s == 0);

    float fv_ = uv.fv();

    float rdelta = r2_ - r1_ ; // 0 for cylinder 
    float zdelta = z2_ - z1_ ;

    assert( zdelta > 0.f );
    zdelta *= 0.95 ; // KLUDGE  

    float r_ = r1_ + rdelta*fv_ ; 
    float z_ = z1_ + zdelta*fv_ ; 

    pos.z = z_  ;

    bool seamed = true ; 
    float azimuth = uv.fu2pi(seamed); 

    float ca = cosf(azimuth);
    float sa = sinf(azimuth);

    pos += glm::vec3( r_*ca, r_*sa, 0.f );
}











/*
npart ncone::part() const 
{
    npart p = nnode::part();
    assert( p.getTypeCode() == CSG_CONE );
    return p ; 
}
*/





/*


    322 // Cone with correct distances to tip and base circle. Y is up, 0 is in the middle of the base.
    323 float fCone(vec3 p, float radius, float height) {
    324     vec2 q = vec2(length(p.xz), p.y);
    325     vec2 tip = q - vec2(0, height);              // cone coordinates (radial, axial) from apex
    326     vec2 mantleDir = normalize(vec2(height, radius));
    327     float mantle = dot(tip, mantleDir);
    328     float d = max(mantle, -q.y);
    329     float projected = dot(tip, vec2(mantleDir.y, -mantleDir.x));
    330 
    331     // distance to tip
    332     if ((q.y > height) && (projected < 0)) {
    333         d = max(d, length(tip));
    334     }
    335 
    336     // distance to base ring
    337     if ((q.x > radius) && (projected > length(vec2(height, radius)))) {
    338         d = max(d, length(q - vec2(radius, 0)));
    339     }
    340     return d;
    341 }





See env-;sdf-

http://iquilezles.org/www/articles/distfunctions/distfunctions.htm

::

    Cone - signed - exact

    float sdCone( vec3 p, vec2 c )
    {
        // c must be normalized
        float q = length(p.xy);          // perpendicular distance of p from axis 
        return dot(c,vec2(q,p.z));       //  c (radius,height) normalized
    }



    Infinite Cone - exact

    float sdCone( vec3 p, vec2 c )
    {
        // c is the sin/cos of the angle

        vec2 q = vec2( length(p.xz), -p.y );         (radial, vertical) query point 

        float d = length(q-c*max(dot(q,c), 0.0));    

        return d * ((q.x*c.y-q.y*c.x<0.0)?-1.0:1.0);

        // 


    }

          

                   tan(th) = r/h  sin/cos
                         /
                +       +  (r,h)
         \      :      /
          \     :     /
           \    :    C                   OC [r, h]
            \   :   /   .                CQ [h, -r]
             \  :  /         Q 
              \ : /     .    
               \:/  .
                O                        BQ [h, r]
               / \
              /   B
             /     \






    Capped Cone - signed - bound

    float sdCappedCone( in vec3 p, in vec3 c )
    {
        vec2 q = vec2( length(p.xz), p.y );    // (perp dist from axis, dist along axis) 
        vec2 v = vec2( c.z*c.y/c.x, -c.z );    //  
        vec2 w = v - q;
        vec2 vv = vec2( dot(v,v), v.x*v.x );
        vec2 qv = vec2( dot(v,w), v.x*w.x );
        vec2 d = max(qv,0.0)*qv/vv;
        return sqrt( dot(w,w) - max(d.x,d.y) ) * sign(max(q.y*v.x-q.x*v.y,w.y));
    }



https://github.com/marklundin/glsl-sdf-primitives/blob/master/sdCappedCone.glsl

::

    float sdCone( in vec3 p, in vec3 c )
    {
        vec2 q = vec2( length(p.xz), p.y );   // query point in (radial, vertical) coordinates  (assume apex origin) 

        float d1 = -p.y-c.z;                  // vertical distance to cap 



        float d2 = max( dot(q,c.xy), p.y);


        return length(max(vec2(d1,d2),0.0)) + min(max(d1,d2), 0.);
    }


http://mercury.sexy/hg_sdf/

sdf-edit::

    322 // Cone with correct distances to tip and base circle. Y is up, 0 is in the middle of the base.
    323 float fCone(vec3 p, float radius, float height) {
    324     vec2 q = vec2(length(p.xz), p.y);    

            // express query point q in (radial,vertical) coordinates 


    325     vec2 tip = q - vec2(0, height);              // cone coordinates (radial, axial) from apex

             apex is radius 0 : curious why not use apex as the origin of operations   ?
            
             tip is vector : TQ
      




                                (0,h) r
                   +------------T------------A - - - - - 
                   :\           :           / 
                   : \          :          /        OA = [ r, h ]
                   :  \         :         /   
                   :   \        L.       /    
                   :    \       :   .   /     
                   :     \      :      I        OIQ : right angle      OI.IQ = 0   perpendicular normal
                   :      \     :     /   .                           
                   :       \    :    /       .                 [ r, h ].[ h, -r ] = 0
                   :        \   :   /           .  
                   :         \  :  /               .         IQ [ h, -r]
                   :          \ : /                   . 
                   :           \:/                      Q 
                 - :- - - - - - O - - - - - - - -   . - - - - - - - 
                   :           /:\               .
                   :          / : \           .
                   :         /  :  \       .
                   :        /   :   \   .
                   :       /    :    J         JQ [ h, r]
                   :      /     :  .  \
                   :     /      M      \
                   :    /       :       \
                   :   /        :        \
                   :  /         :         \
                   : /          :          \
                   ;/           :           \
                   +------------S------------C
                                      r


    326     vec2 mantleDir = normalize(vec2(height, radius));

                //  JQ [h, r]  normalized


    327     float mantle = dot(tip, mantleDir);

                //  distance to the lower sheet 


    328     float d = max(mantle, -q.y);





    329     float projected = dot(tip, vec2(mantleDir.y, -mantleDir.x));
    330 
    331     // distance to tip
    332     if ((q.y > height) && (projected < 0)) {
    333         d = max(d, length(tip));
    334     }
    335 
    336     // distance to base ring
    337     if ((q.x > radius) && (projected > length(vec2(height, radius)))) {
    338         d = max(d, length(q - vec2(radius, 0)));
    339     }
    340     return d;
    341 }




http://aka-san.halcy.de/distance_fields_prefinal.pdf
~/opticks_refs/Procedural_Modelling_with_Signed_Distance_Functions_Thesis.pdf


    A cone can be thought of as a cylinder for which the radius slowly decreases
    towards one end. Starting from this, the radius of a cone around the y-axis with
    opening angle theta at height py can be calculated, yielding r = |py| tan(theta).
    Using this, we can determine the length of the hypotenuse of the triangle spanned by
    the cones side, the line parallel to the xz-plane from the cone to P, and the line
    from P orthogonal to the cones side (i.e. the line of minimal length from P to
    the cone). Trigonometric identities and simplification then give the distance of
    P from the infinite cone. As it is advantageous from a user-interface perspective
    to have an input for base radius and height of cone instead of opening angle,
    DFModel calculates theta from these as theta = atan( r/h ). 

    To make the cone finite, it is again intersected with an y-oriented 
    infinite slab of height h.

    theta = atan( r/h ) 

    d_cone(p,r,h)= max( sqrt(px*px +pz*pz)*cos(theta) - abs(py)sin(theta) , py-h , -py ) 


    As with the cylinder, the intersection makes this function return a distance
    estimate rather than an exact distance.

*/


