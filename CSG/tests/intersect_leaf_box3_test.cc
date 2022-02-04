/**
intersect_leaf_box3_test.cc
===================================

This is for very low level testing of csg_intersect_node.h intersect functions.
For a slightly higher level test see CSGNodeScanTest.cc

**/

#include "OPTICKS_LOG.hh"
#include <cmath>


#include "scuda.h"
#include "squad.h"
#include "sqat4.h"

#define DEBUG 1 
#include "csg_intersect_leaf.h"
//#include "csg_intersect_node.h"
//#include "csg_intersect_tree.h"



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    quad q0 ; 
    q0.f.x = 2.f*5.f ;  
    q0.f.y = 2.f*10.f ; 
    q0.f.z = 2.f*15.f ; 
    q0.f.w = 0.f ; 
    
    float4 isect = make_float4(0.f, 0.f, 0.f, 0.f ); 
    float t_min = 0.f ; 

    float3 ray_origin    = make_float3(  0.f, 0.f, 0.f ); 
    float3 ray_direction = make_float3(  0.f, 0.f, 1.f ); 
 
    bool valid_isect = intersect_leaf_box3( isect, q0, t_min, ray_origin, ray_direction ); 

    float3 pos = make_float3( 0.f, 0.f, 0.f ); 
    if(valid_isect)
    {
        float t = isect.w ; 
        pos = ray_origin + t*ray_direction ; 
        float sd = distance_leaf_box3(pos, q0) ; 

        printf("//pos %10.4f %10.4f %10.4f    sd %10.4f \n", pos.x, pos.y, pos.z, sd ); 
    }

    return 0 ; 
}



