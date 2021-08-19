#include "sutil_vec_math.h"

#include "qat4.h"
#include "csg_intersect_node.h"
#include "csg_intersect_tree.h"

#include "CSGPrim.h"
#include "CSGNode.h"


#include <optix_device.h>

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(float3, position,         attribute position, );  
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, );  
rtDeclareVariable(unsigned,  intersect_identity,   attribute intersect_identity, );  

rtDeclareVariable(unsigned, identity,  ,);   
// "identity" is planted into pergi["identity"] 


// per solid context, not global 
rtBuffer<CSGPrim> prim_buffer;


// global context
rtBuffer<CSGNode> node_buffer;
rtBuffer<qat4> itra_buffer;
rtBuffer<float4> plan_buffer;



/**
As the primIdx argument is in 0:num_prim-1 need separate prim_buffer per geometry 
unlike nodes and itra where s context level node_buffer and itra_buffer allows 
the pre-7 machinery to more closely match optix7
**/

RT_PROGRAM void intersect(int primIdx)
{
    const CSGPrim* prim = &prim_buffer[primIdx] ;   
    int nodeOffset = prim->nodeOffset() ;  
    int numNode = prim->numNode() ; 
    const CSGNode* node = &node_buffer[nodeOffset] ; 
    const float4* plan = &plan_buffer[0] ;  
    const qat4*   itra = &itra_buffer[0] ;  

    float4 isect ; 
    if(intersect_prim(isect, numNode, node, plan, itra, ray.tmin , ray.origin, ray.direction ))
    {
        if(rtPotentialIntersection(isect.w))
        {
            position = ray.origin + isect.w*ray.direction ;   
            shading_normal = make_float3( isect.x, isect.y, isect.z ); 
            intersect_identity = (( (1u+primIdx) & 0xff ) << 24 ) | ( identity & 0x00ffffff ) ; 
            rtReportIntersection(0);
        }
    }
}

RT_PROGRAM void bounds (int primIdx, float result[6])
{
    const CSGPrim* prim = &prim_buffer[primIdx] ; 
    int nodeOffset = prim->nodeOffset() ;  
    int numNode = prim->numNode() ; 
    const float* aabb = prim->AABB();  

    result[0] = *(aabb+0); 
    result[1] = *(aabb+1); 
    result[2] = *(aabb+2); 
    result[3] = *(aabb+3); 
    result[4] = *(aabb+4); 
    result[5] = *(aabb+5); 

#ifdef DEBUG
    rtPrintf("// bounds identity %d primIdx %d nodeOffset %d numNode %d aabb %10.3f %10.3f %10.3f   %10.3f %10.3f %10.3f  \n", 
         identity, primIdx, nodeOffset, numNode,
         result[0], result[1], result[2],  
         result[3], result[4], result[5] 
        ); 
#endif


}


