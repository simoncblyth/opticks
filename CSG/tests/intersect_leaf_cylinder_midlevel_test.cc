/**
intersect_leaf_cylinder_midlevel_test.cc
=========================================

**/

#include "OPTICKS_LOG.hh"
#include <cmath>


#include "scuda.h"
#include "squad.h"
#include "sqat4.h"

#define DEBUG 1 
#include "csg_intersect_leaf.h"




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

    midlevel_test(); 

    return 0 ; 
}



