#include <stdio.h>
#include <optix.h>

#include "scuda.h"
#include "squad.h"
#include "qat4.h"

// simulation 
#include <curand_kernel.h>
#include "qsim.h"
#include "qevent.h"

#include "csg_intersect_node.h"
#include "csg_intersect_tree.h"

#include "Binding.h"
#include "Params.h"


extern "C" { __constant__ Params params ;  }


/**
trace : pure function, with no use of params, everything via args
-------------------------------------------------------------------

optixTrace ray_origin ray_direction args passed by float3 value, 
but could use reference args float4& here and make_float3 from them 

**/

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
        )   // pure 
{
    uint32_t p0, p1, p2, p3 ;
    uint32_t p4, p5, p6, p7 ;

    // hmm: why ? no need to input these ?
    p0 = float_as_uint( normal->x );
    p1 = float_as_uint( normal->y );
    p2 = float_as_uint( normal->z );
    p3 = float_as_uint( *t );

    p4 = float_as_uint( position->x );
    p5 = float_as_uint( position->y );
    p6 = float_as_uint( position->z );
    p7 = *identity ;
 
    const unsigned SBToffset = 0u ; 
    const unsigned SBTstride = 1u ; 
    const unsigned missSBTIndex = 0u ; 
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

/**
*setPayload* is used from __closesthit__ and __miss__ resulting 
in the optixTrace output argument uints converted back to float above.
Notice that could squeeze the payload in half by not including 
the position and t. This is possible as the t output argument allows 
to recalculate intersect position from from ray_position, ray_direction.
**/
static __forceinline__ __device__ void setPayload( float3 normal, float t, float3 position, unsigned identity ) // pure? 
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

__forceinline__ __device__ uchar4 make_color( const float3& normal, unsigned identity )  // pure 
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




/**
render : non-pure, uses params for viewpoint inputs and pixels output 
-----------------------------------------------------------------------

**/

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

    float3 diddled_normal = normalize(normal)*0.5f + 0.5f ; // lightens render, with mid-grey "pedestal" 
    uchar4 color = make_color( diddled_normal, identity );
    unsigned index = idx.y * params.width + idx.x ;

    params.pixels[index] = color ; 
    params.isect[index] = make_float4( position.x, position.y, position.z, uint_as_float(identity)) ; 
}
 
/**
simulate : uses params for input: gensteps, seeds and output photons 
----------------------------------------------------------------------

**/

static __forceinline__ __device__ void simulate( const uint3& idx, const uint3& dim )
{
    qevent* evt      = params.evt ; 
    if (idx.x >= evt->num_photon) return;

    unsigned photon_id = idx.x ; 
    unsigned genstep_id = evt->seed[photon_id] ; 
    const quad6& gs     = evt->genstep[genstep_id] ; 

/*
    printf("//cx.simulate photon_id %3d genstep_id %3d  gs.q0.i ( %3d %3d %3d %3d ) \n", 
       photon_id, 
       genstep_id, 
       gs.q0.i.x, 
       gs.q0.i.y,
       gs.q0.i.z, 
       gs.q0.i.w
      ); 
*/  
      
    qsim<float>* sim = params.sim ; 
    curandState rng = sim->rngstate[photon_id] ; 
    // TODO: skipahead using an event_id 
    quad4 p ;   
    sim->generate_photon(p, rng, gs, photon_id, genstep_id );  

    float3 origin    = make_float3( p.q0.f.x, p.q0.f.y, p.q0.f.z ) ; 
    float3 direction = make_float3( p.q1.f.x, p.q1.f.y, p.q1.f.z ) ; 

    float    t = 0.f ; 
    float3   normal   = make_float3( 0.5f, 0.5f, 0.5f );
    float3   position = make_float3(  0.f, 0.f, 0.f );
    unsigned identity = 0u ; 

    // CONSIDER: ox,oy,oz,px,py,pz args so can avoid origin, direction
    // or directly use a floa44& arguments so can p.q0 p.q1  
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

    p.q0.f.x = position.x ; 
    p.q0.f.y = position.y ; 
    p.q0.f.z = position.z ; 
    p.q0.f.w = 101.f ; 

    p.q1.f.x = direction.x ; 
    p.q1.f.y = direction.y ; 
    p.q1.f.z = direction.z ; 
    p.q1.f.w = 202.f ; 

    p.q2.f.x = params.tmin ; 
    p.q2.f.y = params.tmax ; 
    p.q2.f.z = t ; 
    p.q2.f.w = 303.f ; 

    p.q3.f.x = normal.x ; 
    p.q3.f.y = normal.y ; 
    p.q3.f.z = normal.z ; 
    p.q3.u.w = identity ; 

    evt->photon[photon_id] = p ; 
}

/**
**/


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

    float4 isect ; // .xyz normal .w distance  
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


/**
optix_7_device.h::

    516 ///
    517 /// For a given OptixBuildInputCustomPrimitiveArray the number of primitives is defined as
    518 /// numAabbs.  The primitive index returns is the index into the corresponding build array
    519 /// plus the primitiveIndexOffset.
    520 ///
    521 /// In Intersection and AH this corresponds to the currently intersected primitive.
    522 /// In CH this corresponds to the primitive index of the closest intersected primitive.
    523 /// In EX with exception code OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_HIT_SBT corresponds to the active primitive index. Returns zero for all other exceptions.
    524 static __forceinline__ __device__ unsigned int optixGetPrimitiveIndex();
    525 

**/

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
    unsigned prim_id  = 1u + optixGetPrimitiveIndex() ;  // see GAS_Builder::MakeCustomPrimitivesBI_11N  (1+index-of-CSGPrim within CSGSolid/GAS)

    unsigned identity = (( prim_id & 0xff ) << 24 ) | ( instance_id & 0x00ffffff ) ; 

    // no GAS id ?
    //
    //    0xffffff : 16 M    wasted byte 
    //    0x00ffff : 65535   more reasonable max instances 
    //
    // TODO: use primitiveIndexOffset to encode the solid idx in high bytes 
    //
    // global is problematic, as lots of prim in the one GAS and only one instance_id
    // -> use one bit to switch between two packing schemes 
    // hmm but how to distinguish, as instance_id can be zero ... 1-base it somehow?
    //

    float3 normal = optixTransformNormalFromObjectToWorldSpace( isect_normal ) ;  
    // pre-diddling normal for rendering purposes not acceptable when using for both rendering and simulation   

    const float3 world_origin = optixGetWorldRayOrigin() ; 
    const float3 world_direction = optixGetWorldRayDirection() ; 
    const float3 world_position = world_origin + t*world_direction ; 

    setPayload( normal, t,  world_position, identity );  // communicate to raygen 
}

extern "C" __global__ void __miss__ms()
{
    MissData* ms  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    float3 normal = make_float3( ms->r, ms->g, ms->b );   // this is render specific pre-diddling too
    float t_cand = 0.f ; 
    float3 position = make_float3( 0.f, 0.f, 0.f ); 
    unsigned identity = 0u ; 
    setPayload( normal,  t_cand, position, identity );
}

