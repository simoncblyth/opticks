// ./csg_intersect_leaf_test.sh

#include <cstdio>

#include "ssys.h"
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "csg_intersect_leaf.h"

#include "CSGNode.h"

void test_intersect_leaf_Sphere()
{
    CSGNode nd = CSGNode::Sphere(100.f); 

    const CSGNode* node = &nd ; 
    const float4* plan = nullptr ; 
    const qat4* itra = nullptr ; 
    const float t_min = 0 ; 

    float3 ray_origin    = {0.f, 0.f, -50.f }; 
    float3 ray_direction = {0.f, 0.f, -1.f }; 
    float4 isect = {0.f, 0.f, 0.f, 0.f } ; 

    bool is = intersect_leaf(isect, node, plan, itra, t_min, ray_origin, ray_direction ); 

    printf("// is = %d ; isect = np.array([%10.5f,%10.5f,%10.5f,%10.5f]) \n", is, isect.x, isect.y, isect.z, isect.w ) ;  
}

int main()
{
    test_intersect_leaf_Sphere() ; 

    return 0 ; 
}
