/**
CSGIntersectTest.cc
=====================

This is for very low level testing of csg_intersect_node.h intersect functions.
For a slightly higher level test see CSGNodeScanTest.cc

**/

#include "OPTICKS_LOG.hh"
#include <cmath>


#include "scuda.h"
#include "squad.h"
#include "sqat4.h"

#define DEBUG 1 
#include "csg_intersect_node.h"
#include "csg_intersect_tree.h"



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    double startPhi = 0.25 ; 
    double deltaPhi = 0.1 ; 

    double phi0 = startPhi ; 
    double phi1 = startPhi + deltaPhi  ; 

    double pi = M_PI ; 
    double cosPhi0 = cos(phi0*pi ); 
    double sinPhi0 = sin(phi0*pi ); 
    double cosPhi1 = cos(phi1*pi ); 
    double sinPhi1 = sin(phi1*pi ); 

    quad q0 ; 
    q0.f.x = cosPhi0 ;  
    q0.f.y = sinPhi0 ; 
    q0.f.z = cosPhi1 ; 
    q0.f.w = sinPhi1 ; 
    
    float4 isect = make_float4(0.f, 0.f, 0.f, 0.f ); 
    float t_min = 0.f ; 

    float3 ray_origin    = make_float3( 10.f, 0.f, 0.f ); 
    float3 ray_direction = make_float3(  0.f, 1.f, 0.f ); 
 
    bool valid_isect = intersect_node_phicut( isect, q0, t_min, ray_origin, ray_direction ); 
    float4 post = make_float4( 0.f , 0.f, 0.f , 0.f ); 

    if(valid_isect)
    {
        float t = isect.w ; 
        post.x = ray_origin.x + t*ray_direction.x ; 
        post.y = ray_origin.y + t*ray_direction.y ; 
        post.z = ray_origin.z + t*ray_direction.z ; 
        post.w = t ; 

        printf("//post %10.4f %10.4f %10.4f %10.4f \n", post.x, post.y, post.z, post.w ); 
    }

    return 0 ; 
}



