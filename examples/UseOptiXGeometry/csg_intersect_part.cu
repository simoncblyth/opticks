/**
csg_intersect_part.cu
=======================

Attempt to make the Opticks CSG primitives testable in a standalone-ish manner.

NB this is independent of the lower level csg_intersect_prim.cu 

**/

#include "OpticksCSG.h"

#include <optix_world.h>
using namespace optix;

#include "math_constants.h"   // CUDART_ defines

#include "cu/quad.h"
#include "cu/Part.h"
#include "cu/bbox.h"
#include "cu/csg_intersect_primitive.h"
#include "cu/Prim.h"

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtBuffer<Part> partBuffer;
rtBuffer<Matrix4x4> tranBuffer;
//rtBuffer<Prim>  primBuffer;

#include "cu/postorder.h"
#include "cu/csg_intersect_part.h"

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 


static __device__
void dummy_prim(Prim& prim) 
{
    int part_offset = 0 ; 
    int num_parts = 1 ; 
    int tran_offset = 0 ; 
    int plan_offset = 0 ; 
    prim.q0.i = make_int4( part_offset, num_parts, tran_offset, plan_offset ); 
}


RT_PROGRAM void bounds(int primIdx, float result[6])
{
    //const Prim& prim    = primBuffer[primIdx];
    Prim prim ; 
    dummy_prim(prim); 

    optix::Aabb* aabb = (optix::Aabb*)result;
    csg_bounds_prim(primIdx, prim, aabb );
    rtPrintf("//bounds result (%f %f %f) (%f %f %f) \n", result[0], result[1], result[2], result[3], result[4], result[5] );
}

RT_PROGRAM void intersect(int /*primIdx*/)
{
    //const Prim& prim    = primBuffer[primIdx];
    Prim prim ; 
    dummy_prim(prim); 

    unsigned partIdx = 0u ; 

    float tt_min = 0.001f ; 
    float4 tt = make_float4(0.f,0.f,1.f, tt_min);

    csg_intersect_part(prim, partIdx, tt_min, tt  );

    if(rtPotentialIntersection(tt.w))
    {
        shading_normal.x = geometric_normal.x = tt.x ;
        shading_normal.y = geometric_normal.y = tt.y ;
        shading_normal.z = geometric_normal.z = tt.z ;
        rtReportIntersection(0);
    }
}

