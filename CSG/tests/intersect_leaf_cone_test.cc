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
//#define DEBUG 1 
#include "csg_intersect_leaf.h"

int main(int argc, char** argv)
{
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
    printf("// z_apex_cone %10.4f \n", z0 ); 

    float t_min = 120.f ; 
    float3 ray_origin     = make_float3( -100.f, 0.f, -200.f ); 
    float3 ray_direction_ = make_float3(    1.f, 0.f,    2.f ); 
    float3 ray_direction  = normalize(ray_direction_); 
    float4 isect = make_float4(0.f, 0.f, 0.f, 0.f ); 
    float3 pos = make_float3( 0.f , 0.f, 0.f ); 
    float3 zero3 = make_float3( 0.f , 0.f, 0.f ); 

    const float3& o = ray_origin ;  
    const float3& d = ray_direction ;  
    float4& i = isect ; 
    float3& p = pos ; 
    const float& t = isect.w ; 

    bool valid_isect = intersect_leaf_cone( i, q0, t_min, o, d ); 
    p = valid_isect ? o + t*d : zero3  ; 

    printf("// %8s (%10.4f %10.4f %10.4f ; %10.4f %10.4f %10.4f ; %10.4f) valid_isect %d\n// %8s (%10.4f %10.4f %10.4f %10.4f)\n// %8s (%10.4f %10.4f %10.4f)  \n", 
         "ray",
         o.x, o.y, o.z, 
         d.x, d.y, d.z,
         t_min, 
         valid_isect,
         "isect", 
         i.x, i.y, i.z, i.w, 
         "pos",
         p.x, p.y, p.z 
        ); 

    return 0 ; 
}


