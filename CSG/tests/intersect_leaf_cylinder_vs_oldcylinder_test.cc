/**
intersect_leaf_cylinder_test.cc
===================================

**/

#include "OPTICKS_LOG.hh"
#include <cmath>


#include "scuda.h"
#include "squad.h"
#include "sqat4.h"

#define DEBUG 1 
#include "csg_intersect_leaf_head.h"
#include "csg_robust_quadratic_roots.h"

#include "csg_intersect_leaf_cylinder.h"
#include "csg_intersect_leaf_oldcylinder.h"


void lowlevel_test()
{
    quad q0, q1 ; 
    q0.f.x = 0.f ;  
    q0.f.y = 0.f ; 
    q0.f.z = 0.f ; 
    q0.f.w = 10.f ;  // radius
    
    q1.f.x = -5.f ;  // z1 
    q1.f.y =  5.f ;  // z2 
    q1.f.z =  0.f ; 
    q1.f.w =  0.f ; 


    float t_min = 0.f ; 
    float3 ray_origin    = make_float3(  0.f, 0.f, 0.f ); 
    float3 ray_direction = make_float3(  0.f, 0.f, 1.f ); 

    float4 isect0 = make_float4(0.f, 0.f, 0.f, 0.f ); 
    float4 isect1 = make_float4(0.f, 0.f, 0.f, 0.f ); 

    bool valid_isect0(false) ;  
    intersect_leaf_cylinder( valid_isect0, isect0, q0, q1, t_min, ray_origin, ray_direction ); 
    bool valid_isect1(false) ;  
    intersect_leaf_oldcylinder( valid_isect1, isect1, q0, q1, t_min, ray_origin, ray_direction ); 



    //float3 pos0 = make_float3( 0.f , 0.f, 0.f ); 
    //float3 pos1 = make_float3( 0.f , 0.f, 0.f ); 

    if(valid_isect0 && valid_isect1 )
    {
        float t0 = isect0.w ; 
        float t1 = isect1.w ; 

        printf("// t0 %10.4f t1 %10.4f \n", t0, t1 ); 

        /*
        pos0  = ray_origin + t0*ray_direction  ; 
        pos1  = ray_origin + t1*ray_direction  ; 

        float sd0 = distance_leaf_cylinder(pos0, q0, q1) ; 
        float sd1 = distance_leaf_cylinder(pos1, q0, q1) ; 
        */

    }

}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    lowlevel_test(); 


    return 0 ; 
}



