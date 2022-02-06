/**
intersect_leaf_thetacut_test.cc
===================================

This is for very low level testing of csg_intersect_leaf.h intersect functions.
For a slightly higher level test see CSGNodeScanTest.cc

**/

#include "OPTICKS_LOG.hh"
#include <cmath>

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"

#define DEBUG 1 
#include "csg_intersect_leaf.h"


bool intersect( float4& is, float4& p, const quad& q0, char imp, float t_min, const float3& o, const float3& d )
{
    bool valid_isect = false ; 
    switch(imp)
    {
        case ' ': valid_isect = intersect_leaf_thetacut(       is, q0, t_min, o, d )  ;   break ; 
        case 'L': valid_isect = intersect_leaf_thetacut_lucas( is, q0, t_min, o, d )  ;   break ; 
    }

    if(valid_isect)
    {
        float t = is.w ; 
        p.x = o.x + t*d.x ; 
        p.y = o.y + t*d.y ; 
        p.z = o.z + t*d.z ; 
        p.w = t ; 

    }

    if( valid_isect )
    {
        printf("// %c o (%10.4f %10.4f %10.4f)  d (%10.4f %10.4f %10.4f)   p (%10.4f %10.4f %10.4f %10.4f) \n", imp, 
                   o.x, o.y, o.z, d.x, d.y, d.z, p.x, p.y, p.z, p.w ); 
    }
    else
    {
        printf("// %c o (%10.4f %10.4f %10.4f)  d (%10.4f %10.4f %10.4f)   p (MISS) \n", imp, 
                   o.x, o.y, o.z, d.x, d.y, d.z ); 
    }

    return valid_isect ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    float startTheta = 0.25 ; 
    float deltaTheta = 0.5 ; 

    quad q0 ; 
    CSGNode::PrepThetaCutParam( q0, startTheta, deltaTheta ); 
    
    float4 isect = make_float4(0.f, 0.f, 0.f, 0.f ); 
    float4 post  = make_float4(0.f, 0.f, 0.f, 0.f ); 
    float t_min = 0.f ; 

    float3 ray_origin    = make_float3( 10.f, 0.f, 0.f ); 
    float3 ray_direction = make_float3(  0.f, 0.f, 1.f ); 

    
    printf("// x scan \n"); 
    for(float x=-10.f ; x < 10.1f ; x+=0.1f )
    {
        ray_origin.x = x ; 
        bool i0 = intersect( isect, post, q0, ' ', t_min, ray_origin, ray_direction ); 
        bool i1 = intersect( isect, post, q0, 'L', t_min, ray_origin, ray_direction ); 
        assert( i0 == i1 ); 
    }

    printf("// z scan \n"); 
    ray_origin.x = 0.f ;
    ray_origin.y = 0.f ;
   
    ray_direction.x = 1.f ; 
    ray_direction.y = 0.f ; 
    ray_direction.z = 0.f ; 

    for(float z=-10.f ; z < 10.1f ; z+=0.1f )
    {
        ray_origin.z = z ;
        bool i0 = intersect( isect, post, q0, ' ', t_min, ray_origin, ray_direction ); 
        bool i1 = intersect( isect, post, q0, 'L', t_min, ray_origin, ray_direction ); 
        assert( i0 == i1 ); 
    }





    return 0 ; 
}



