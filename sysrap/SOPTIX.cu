/**
SOPTIX.cu
===========


Functions 
-----------

trace
    populate quad2 prd by call to optixTrace

make_color
    minimal normal "shader"

render
    raygen function : calling trace and "shading" pixels

__raygen__rg
    calls one of the above raygen functions depending on params.raygenmode 

setPayload
    mechanics of communication when not using WITH_PRD

__miss_ms
    default quad2 prd OR payload for rays that miss 

__closesthit__ch
    populate quad2 prd OR payload for rays that intersect    

**/

#include <optix.h>
#include <stdint.h>

#include "scuda.h"
#include "squad.h"

#include "SOPTIX_Binding.h"
#include "SOPTIX_Params.h"

#include "scuda_pointer.h"
#include "SOPTIX_getPRD.h"

extern "C" { __constant__ SOPTIX_Params params ;  }

/**
trace : pure function, with no use of params, everything via args
-------------------------------------------------------------------

Outcome of trace is to populate *prd* by payload and attribute passing.
When WITH_PRD macro is defined only 2 32-bit payload values are used to 
pass the 64-bit  pointer, otherwise more payload and attributes values 
are used to pass the contents IS->CH->RG. 

See __closesthit__ch to see where the payload p0-p7 comes from.
**/

static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        quad2*                 prd
        )   
{
    const float rayTime = 0.0f ; 
    OptixVisibilityMask visibilityMask = 1u  ; 
    OptixRayFlags rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT ;   // OPTIX_RAY_FLAG_NONE 
    const unsigned SBToffset = 0u ; 
    const unsigned SBTstride = 1u ; 
    const unsigned missSBTIndex = 0u ; 
    uint32_t p0, p1 ; 
    packPointer( prd, p0, p1 ); 
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            rayTime,
            visibilityMask,
            rayFlags,
            SBToffset,
            SBTstride,
            missSBTIndex,
            p0, p1
            );
}


__forceinline__ __device__ uchar4 make_color( const float3& normal, unsigned identity, unsigned boundary )  // pure 
{
    // identity and boundary not used 
    float scale = 1.f ; 
    return make_uchar4(
            static_cast<uint8_t>( clamp( normal.x, 0.0f, 1.0f ) *255.0f )*scale ,
            static_cast<uint8_t>( clamp( normal.y, 0.0f, 1.0f ) *255.0f )*scale ,
            static_cast<uint8_t>( clamp( normal.z, 0.0f, 1.0f ) *255.0f )*scale ,
            255u
            );
}



/**
render : non-pure, uses params for viewpoint inputs and pixels output 
-----------------------------------------------------------------------

**/

static __forceinline__ __device__ void render( const uint3& idx, const uint3& dim, quad2* prd )
{
    float2 d = 2.0f * make_float2(
            static_cast<float>(idx.x)/static_cast<float>(dim.x),
            static_cast<float>(idx.y)/static_cast<float>(dim.y)
            ) - 1.0f;

    const bool yflip = true ;
    if(yflip) d.y = -d.y ;

    const unsigned cameratype = params.cameratype ;  
    const float3 dxyUV = d.x * params.U + d.y * params.V ; 
    const float3 origin    = cameratype == 0u ? params.eye                     : params.eye + dxyUV    ;
    const float3 direction = cameratype == 0u ? normalize( dxyUV + params.W )  : normalize( params.W ) ;
    //                           cameratype 0u:perspective,                    1u:orthographic

    trace( 
        params.handle,
        origin,
        direction,
        params.tmin,
        params.tmax,
        prd
    );

    const float3* normal = prd->normal();  
    float3 diddled_normal = normalize(*normal)*0.5f + 0.5f ; 
    // "diddling" changes range of elements from -1.f:1.f to 0.f:1.f same as  (n+1.f)/2.f   
    unsigned index = idx.y * params.width + idx.x ;

    params.pixels[index] = make_color( diddled_normal, prd->identity(), prd->boundary() ); 
}
 

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    quad2 prd ; 
    prd.zero(); 
 
    render(   idx, dim, &prd );
} 

/**
__miss__ms
-------------

* missing "normal" is somewhat render specific and this is used for 
  all raygenmode but Miss should never happen with real simulations 
* Miss can happen with simple geometry testing however when shoot 
  rays from outside the "world"

**/

extern "C" __global__ void __miss__ms()
{
    SOPTIX_MissData* ms = reinterpret_cast<SOPTIX_MissData*>( optixGetSbtDataPointer() );
    const unsigned identity = 0xffffffffu ; 
    const unsigned boundary = 0xffffu ;
    const float lposcost = 0.f ; 
  
    quad2* prd = getPRD<quad2>(); 

    prd->q0.f.x = ms->bg_color.x ;   
    prd->q0.f.y = ms->bg_color.y ; 
    prd->q0.f.z = ms->bg_color.z ; 
    prd->q0.f.w = 0.f ; 

    prd->q1.u.x = 0u ; 
    prd->q1.u.y = 0u ; 
    prd->q1.u.z = 0u ; 
    prd->q1.u.w = 0u ; 

    prd->set_boundary(boundary); 
    prd->set_identity(identity); 
    prd->set_lposcost(lposcost); 
}

/**
__closesthit__ch 
=================

optixGetInstanceIndex (aka iindex)
    0-based index within IAS

optixGetInstanceId (aka identity)
    user supplied instanceId, 

optixGetPrimitiveIndex (aka prim_idx)
    CustomPrimitiveArray: local index of AABB within the GAS, 
    TriangleArray: local index of triangle (HMM: within one buildInput?)

optixGetRayTmax
    In intersection and CH returns the current smallest reported hitT or the tmax passed into rtTrace 
    if no hit has been reported

**/

extern "C" __global__ void __closesthit__ch()
{
    // in general will need to branch between 
    //      OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES
    //      OPTIX_BUILD_INPUT_TYPE_TRIANGLES
    // here just assume triangles

    const SOPTIX_HitgroupData* hit_group_data = reinterpret_cast<SOPTIX_HitgroupData*>( optixGetSbtDataPointer() );
    const SOPTIX_TriMesh& mesh = hit_group_data->mesh ; 

    const unsigned prim_idx = optixGetPrimitiveIndex();
    const float2   barys    = optixGetTriangleBarycentrics();

    uint3 tri = mesh.indice[ prim_idx ];
    const float3 P0 = mesh.vertex[ tri.x ];
    const float3 P1 = mesh.vertex[ tri.y ];
    const float3 P2 = mesh.vertex[ tri.z ];

    const float3 P = ( 1.0f-barys.x-barys.y)*P0 + barys.x*P1 + barys.y*P2;
    const float3 Ng = cross( P1-P0, P2-P0 );
    const float3 N = normalize( optixTransformNormalFromObjectToWorldSpace( Ng ) );
    // HMM: could get normal by bary-weighting vertex normals ?

    unsigned iindex = optixGetInstanceIndex() ; 
    unsigned identity = optixGetInstanceId() ;  
    unsigned boundary = 0 ; 
    // HMM: need to plant boundary in HitGroupData ? 
    // cf CSGOptiX/Analytic: node->boundary();// all nodes of tree have same boundary 

    float t = optixGetRayTmax() ;

    // cannot get ray_origin/direction in CH (only IS,AH)
    //const float3 ray_origin = optixGetObjectRayOrigin();
    //const float3 ray_direction = optixGetObjectRayDirection();
    //const float3 lpos = ray_origin + t*ray_direction  ; 
    // HMM: could use P to give the local position ?  

    float lposcost = normalize_z(P); // scuda.h 

    quad2* prd = getPRD<quad2>(); 

    prd->q0.f.x = N.x ;  
    prd->q0.f.y = N.y ;  
    prd->q0.f.z = N.z ;  
    prd->q0.f.w = t ;  

    prd->set_identity( identity ) ;
    prd->set_iindex(   iindex ) ;
    prd->set_boundary(boundary) ; 
    prd->set_lposcost(lposcost); 
}

