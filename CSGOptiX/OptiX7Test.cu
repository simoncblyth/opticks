#include <stdio.h>
#include <optix.h>

#include "sutil_vec_math.h"
#include "qat4.h"

#include "csg_intersect_node.h"
#include "csg_intersect_tree.h"

#include "Binding.h"
#include "Params.h"


extern "C" { __constant__ Params params ;  }

static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        float3*                normal, 
        float*                 t, 
        float3*                position,
        unsigned*              identity
        )
{
    uint32_t p0, p1, p2, p3 ;
    uint32_t p4, p5, p6, p7 ;

    p0 = float_as_uint( normal->x );
    p1 = float_as_uint( normal->y );
    p2 = float_as_uint( normal->z );
    p3 = float_as_uint( *t );

    p4 = float_as_uint( position->x );
    p5 = float_as_uint( position->y );
    p6 = float_as_uint( position->z );
    p7 = *identity ;
 
    unsigned SBToffset = 0u ; 
    unsigned SBTstride = 1u ; 
    unsigned missSBTIndex = 0u ; 
    const float rayTime = 0.0f ; 

    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            rayTime,
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            SBToffset,
            SBTstride,
            missSBTIndex,
            p0, p1, p2, p3, 
            p4, p5, p6, p7
            );

    normal->x = uint_as_float( p0 );
    normal->y = uint_as_float( p1 );
    normal->z = uint_as_float( p2 );
    *t        = uint_as_float( p3 ); 

    position->x = uint_as_float( p4 );
    position->y = uint_as_float( p5 );
    position->z = uint_as_float( p6 );
    *identity   = p7 ; 
 
}

static __forceinline__ __device__ void setPayload( float3 normal, float t, float3 position, unsigned identity )
{
    optixSetPayload_0( float_as_uint( normal.x ) );
    optixSetPayload_1( float_as_uint( normal.y ) );
    optixSetPayload_2( float_as_uint( normal.z ) );
    optixSetPayload_3( float_as_uint( t ) );

    optixSetPayload_4( float_as_uint( position.x ) );
    optixSetPayload_5( float_as_uint( position.y ) );
    optixSetPayload_6( float_as_uint( position.z ) );
    optixSetPayload_7( identity );
}

__forceinline__ __device__ uchar4 make_color( const float3& normal, unsigned identity )
{
    //float scale = iidx % 2u == 0u ? 0.5f : 1.f ; 
    float scale = 1.f ; 
    return make_uchar4(
            static_cast<uint8_t>( clamp( normal.x, 0.0f, 1.0f ) *255.0f )*scale ,
            static_cast<uint8_t>( clamp( normal.y, 0.0f, 1.0f ) *255.0f )*scale ,
            static_cast<uint8_t>( clamp( normal.z, 0.0f, 1.0f ) *255.0f )*scale ,
            255u
            );
}



static __forceinline__ __device__ void render( const uint3& idx, const uint3& dim )
{
    float2 d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
            ) - 1.0f;

    const bool yflip = true ;
    if(yflip) d.y = -d.y ;

    const unsigned cameratype = params.cameratype ;  
    const float3 dxyUV = d.x * params.U + d.y * params.V ; 
    //                           cameratype 0u:perspective,                    1u:orthographic
    const float3 origin    = cameratype == 0u ? params.eye                     : params.eye + dxyUV    ;
    const float3 direction = cameratype == 0u ? normalize( dxyUV + params.W )  : normalize( params.W ) ;

    float    t = 0.f ; 
    float3   normal   = make_float3( 0.5f, 0.5f, 0.5f );
    float3   position = make_float3(  0.f, 0.f, 0.f );
    unsigned identity = 0u ; 

    trace( 
        params.handle,
        origin,
        direction,
        params.tmin,
        params.tmax,
        &normal, 
        &t, 
        &position,
        &identity
    );

    uchar4 color = make_color( normal, identity );
    unsigned index = idx.y * params.width + idx.x ;

    params.pixels[index] = color ; 
    params.isect[index] = make_float4( position.x, position.y, position.z, uint_as_float(identity)) ; 
}
 

static __forceinline__ __device__ void simulate( const uint3& idx, const uint3& dim )
{
    // generate single photon from input params.gensteps[0]  
    // propagate photon 
    printf("//simulate idx.x %d \n", idx.x ); 
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    switch( params.raygenmode )
    {
        case 0: render(   idx, dim ) ; break ;  
        case 1: simulate( idx, dim ) ; break ;  
    }
} 


extern "C" __global__ void __miss__ms()
{
    MissData* ms  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    float3 normal = make_float3( ms->r, ms->g, ms->b );   
    float t_cand = 0.f ; 
    float3 position = make_float3( 0.f, 0.f, 0.f ); 
    unsigned identity = 0u ; 
    setPayload( normal,  t_cand, position, identity );
}

/**
__intersection__is
----------------------

HitGroupData provides the numNode and nodeOffset of the intersected CSGPrim.
Which Prim gets intersected relies on the CSGPrim::setSbtIndexOffset

**/
extern "C" __global__ void __intersection__is()
{
    HitGroupData* hg  = (HitGroupData*)optixGetSbtDataPointer();  
    int numNode = hg->numNode ;      
    int nodeOffset = hg->nodeOffset ; 

    const CSGNode* node = params.node + nodeOffset ;  
    const float4* plan = params.plan ;  
    const qat4*   itra = params.itra ;  

    const float  t_min = optixGetRayTmin() ; 
    const float3 ray_origin = optixGetObjectRayOrigin();
    const float3 ray_direction = optixGetObjectRayDirection();

    float4 isect ; 
    if(intersect_prim(isect, numNode, node, plan, itra, t_min , ray_origin, ray_direction ))
    {
        const unsigned hitKind = 0u ; 
        unsigned a0, a1, a2, a3;      

        a0 = float_as_uint( isect.x );
        a1 = float_as_uint( isect.y );
        a2 = float_as_uint( isect.z );
        a3 = float_as_uint( isect.w ) ; 

        optixReportIntersection( isect.w, hitKind, a0, a1, a2, a3 );
    }
}

extern "C" __global__ void __closesthit__ch()
{
    const float3 isect_normal =
        make_float3(
                uint_as_float( optixGetAttribute_0() ),
                uint_as_float( optixGetAttribute_1() ),
                uint_as_float( optixGetAttribute_2() )
                );
    
    const float t = uint_as_float( optixGetAttribute_3() ) ;  

    unsigned instance_id = optixGetInstanceId() ;        // see IAS_Builder::Build and InstanceId.h 
    unsigned prim_id  = 1u + optixGetPrimitiveIndex() ;  // see GAS_Builder::MakeCustomPrimitivesBI 
    unsigned identity = (( prim_id & 0xff ) << 24 ) | ( instance_id & 0x00ffffff ) ; 

    float3 normal = normalize( optixTransformNormalFromObjectToWorldSpace( isect_normal ) ) * 0.5f + 0.5f ;  

    const float3 world_origin = optixGetWorldRayOrigin() ; 
    const float3 world_direction = optixGetWorldRayDirection() ; 
    const float3 world_position = world_origin + t*world_direction ; 

    setPayload( normal, t,  world_position, identity );
}

