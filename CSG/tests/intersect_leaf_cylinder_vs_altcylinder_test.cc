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
#include "csg_intersect_leaf.h"

#include "CSGNode.h"


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

    bool valid_isect0 = intersect_leaf_cylinder(    isect0, q0, q1, t_min, ray_origin, ray_direction ); 
    bool valid_isect1 = intersect_leaf_oldcylinder( isect1, q0, q1, t_min, ray_origin, ray_direction ); 



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


void midlevel_test()
{
    CSGNode nd = CSGNode::Cylinder( 100.f , -1.f, 1.f ) ; 

    bool valid_isect(false); 
    float4 isect = make_float4( 0.f, 0.f, 0.f, 0.f ); 

    float t_min = 2.f ; 
    float3 ray_origin    = make_float3(  0.f, 0.f, 0.f ); 
    float3 ray_direction = make_float3(  0.f, 0.f, 1.f ); 

    float3& o = ray_origin ; 
    float3& v = ray_direction ; 

    for( float x=-150.f ; x <= 150.f ; x += 10.f )
    {
        o.x = x ; 
        valid_isect = intersect_leaf( isect, &nd , nullptr, nullptr, t_min, o, v ); 

        printf("//midlevel_test: o (%10.4f %10.4f %10.4f) v (%10.4f %10.4f %10.4f) valid_isect %d i ( %10.4f %10.4f %10.4f %10.4f ) \n", 
             o.x, o.y, o.z,
             v.x, v.y, v.z,
             valid_isect, 
             isect.x, isect.y, isect.z, isect.w ); 
    }
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //lowlevel_test(); 
    midlevel_test(); 


    return 0 ; 
}



