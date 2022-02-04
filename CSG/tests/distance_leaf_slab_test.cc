/**
distance_leaf_slab_test.cc
===================================

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

    quad q0, q1 ; 
    q0.f.x = 1.f ;  
    q0.f.y = 0.f ; 
    q0.f.z = 0.f ; 
    q0.f.w = 0.f ;   
    
    q1.f.x = -5.f ;  // z1 
    q1.f.y =  5.f ;  // z2 
    q1.f.z =  0.f ; 
    q1.f.w =  0.f ; 

    float3 pos = make_float3( 0.f , 0.f, 0.f ); 
    float sd = distance_leaf_slab( pos, q0, q1 ); 

    printf("//pos %10.4f %10.4f %10.4f  sd   %10.4f \n", pos.x, pos.y, pos.z, sd ); 

    return 0 ; 
}



