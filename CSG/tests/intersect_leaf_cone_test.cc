/**
intersect_leaf_cone_test.cc
===================================

::

   ./intersect_leaf_cone_test.sh



::
 
   Z 
   |
   +--X 

   .               +                     0
                   |
                   |
   .         +---*-+-----+             -50
                   |
                   |
   .   +------*----+-----------+       -100

   .  -100   -50    0     +50   +100

   .   

   .   *


**/

#include <cmath>
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"

#define DEBUG_CONE 1 

#include "csg_intersect_leaf.h"

int main(int argc, char** argv)
{
    float tmn = scuda::efloat(  "TMIN", 0.f) ; 
    float3 o  = scuda::efloat3( "RAYORI", "-100,0,-200" ); 
    float3 d_ = scuda::efloat3( "RAYDIR", "1,0,2" ); 
    float3 d  = normalize(d_); 

    float r1 =  100.f ; 
    float z1 = -100.f ; 
    float r2 =   50.f ; 
    float z2 =  -50.f ; 

    quad q0 ; 
    q0.f.x = r1 ;  
    q0.f.y = z1 ; 
    q0.f.z = r2 ; 
    q0.f.w = z2 ;  

    float z0 = z_apex_cone( q0 ); 
    printf("// r1 %10.4f z1 %10.4f  r2 %10.4f z2 %10.4f apex z0 %10.4f \n", r1,z1,r2,z2,z0 ); 

    float3 zero3 = make_float3( 0.f , 0.f, 0.f ); 
    float4 i0 = make_float4(0.f, 0.f, 0.f, 0.f ); 
    float4 i1 = make_float4(0.f, 0.f, 0.f, 0.f ); 
    float3 p0 = make_float3( 0.f , 0.f, 0.f ); 
    float3 p1 = make_float3( 0.f , 0.f, 0.f ); 

    const float& t0 = i0.w ; 
    const float& t1 = i1.w ; 

    bool vi0 = intersect_leaf_oldcone( i0, q0, tmn, o, d ); 
    bool vi1 = intersect_leaf_newcone( i1, q0, tmn, o, d ); 

    p0 = vi0 ? o + t0*d : zero3  ; 
    p1 = vi1 ? o + t1*d : zero3  ; 


    printf("// ray (%10.4f %10.4f %10.4f ; %10.4f %10.4f %10.4f ; %10.4f)\n",
         o.x, o.y, o.z, d.x, d.y, d.z, tmn ); 
    printf("// vi0 %d i0 (%10.4f %10.4f %10.4f %10.4f)  p0 (%10.4f %10.4f %10.4f)\n", 
         vi0, i0.x, i0.y, i0.z, i0.w, p0.x, p0.y, p0.z ); 
    printf("// vi1 %d i1 (%10.4f %10.4f %10.4f %10.4f)  p1 (%10.4f %10.4f %10.4f)\n", 
         vi1, i1.x, i1.y, i1.z, i1.w, p1.x, p1.y, p1.z ); 
    return 0 ; 
}


