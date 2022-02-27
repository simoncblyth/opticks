#include <optix.h>

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"

// simulation 
#include <curand_kernel.h>
#include "qsim.h"
#include "qevent.h"

#include "csg_intersect_leaf.h"
#include "csg_intersect_node.h"
#include "csg_intersect_tree.h"

#include "Binding.h"
#include "Params.h"

extern "C" { __constant__ Params params ;  }


/**
trace : pure function, with no use of params, everything via args
-------------------------------------------------------------------

See below __closesthit__ch to see where the payload p0-p7 comes from.

**/

static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        float3*                normal, 
        float*                 t, 
        unsigned*              identity,
        unsigned*              boundary,
        float*                 spare1,
        float*                 spare2
        )   // pure 
{
    const unsigned SBToffset = 0u ; 
    const unsigned SBTstride = 1u ; 
    const unsigned missSBTIndex = 0u ; 
    const float rayTime = 0.0f ; 

    unsigned p0, p1, p2, p3 ; 
    unsigned p4, p5, p6, p7 ; 

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
    *identity = p4 ; 
    *boundary = p5 ; 
    *spare1   = uint_as_float( p6 ); 
    *spare2   = uint_as_float( p7 ); 
    // max of 8, perhaps need f_theta, f_phi ?
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
    unsigned identity = 0u ; 
    unsigned boundary = 0u ; 
    float spare1 = 0.f ; 
    float spare2 = 0.f ; 

    trace( 
        params.handle,
        origin,
        direction,
        params.tmin,
        params.tmax,
        &normal, 
        &t, 
        &identity,
        &boundary,
        &spare1,
        &spare2
    );

    float3 position = origin + t*direction ; 
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
     
    qsim<float>* sim = params.sim ; 
    curandState rng = sim->rngstate[photon_id] ;    // TODO: skipahead using an event_id 
    quad4 p ;   
    sim->generate_photon(p, rng, gs, photon_id, genstep_id );  

    float3 origin    = make_float3( p.q0.f.x, p.q0.f.y, p.q0.f.z ) ; 
    float3 direction = make_float3( p.q1.f.x, p.q1.f.y, p.q1.f.z ) ; 

    float    t = 0.f ; 
    float3   normal   = make_float3( 0.5f, 0.5f, 0.5f );
    unsigned identity = 0u ; 
    unsigned boundary = 0u ; 
    float spare1 = 0.f ; 
    float spare2 = 0.f ; 

    bool do_trace = true ; 

    if( do_trace )
    { 
        trace( 
            params.handle,
            origin,
            direction,
            params.tmin,
            params.tmax,
            &normal, 
            &t, 
            &identity,
            &boundary,
            &spare1,  
            &spare2
        );
    }


    // transform (x,z) intersect position into pixel coordinates (ix,iz)
    float3 ipos = origin + t*direction ; 

    // aim to match the CPU side test isect "photons" from CSG/CSGQuery.cc/CSGQuery::intersect_elaborate

    p.q0.f.x = normal.x ; 
    p.q0.f.y = normal.y ; 
    p.q0.f.z = normal.z ; 
    p.q0.f.w = t  ;  

    p.q1.f.x = ipos.x ; 
    p.q1.f.y = ipos.y ; 
    p.q1.i.z = ipos.z ; 
    p.q1.i.w = 0.f ;   // TODO: sd 

    p.q2.f.x = origin.x ; 
    p.q2.f.y = origin.y ; 
    p.q2.f.z = origin.z ; 
    p.q2.u.w = params.tmin ; 

    p.q3.f.x = direction.x ;   
    p.q3.f.y = direction.y ; 
    p.q3.f.z = direction.z ; 
    p.q3.u.w = identity ; 

    evt->photon[photon_id] = p ; 


    // Compose frames of pixels, isect and "fphoton" within the cegs window
    // using the positions of the intersect "photons".
    // Note that multiple threads may be writing to the same pixel 
    // that is apparently not a problem, just which does it is uncontrolled.

    /*
    unsigned index = iz * params.width + ix ;
    if( index > 0 )
    {
        params.pixels[index] = make_uchar4( 255u, 0u, 0u, 255u) ;
        params.isect[index] = make_float4( ipos.x, ipos.y, ipos.z, uint_as_float(identity)) ; 
        params.fphoton[index] = p ; 
    }
    */

}

/**
float cos_theta = dot(normal,direction);

* cos_theta "sign/orient-ing the boundary" up here in raygen unlike oxrap/cu/closest_hit_propagate.cu,
  avoids having to pass the information from lower level
  
* for angular efficiency need intersection point in object frame to get the angles  

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
*setPayload* is used from __closesthit__ and __miss__ providing communication to __raygen__ optixTrace call
**/
static __forceinline__ __device__ void setPayload( float3 normal, float t, unsigned identity, unsigned boundary, float spare1, float spare2  ) // pure? 
{
    optixSetPayload_0( float_as_uint( normal.x ) );
    optixSetPayload_1( float_as_uint( normal.y ) );
    optixSetPayload_2( float_as_uint( normal.z ) );
    optixSetPayload_3( float_as_uint( t ) );
    optixSetPayload_4( identity );
    optixSetPayload_5( boundary );
    optixSetPayload_6( float_as_uint(spare1) );
    optixSetPayload_7( float_as_uint(spare2) );
    // maximum of 8 payload values configured in PIP::PIP 
}


extern "C" __global__ void __miss__ms()
{
    MissData* ms  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    float3 normal = make_float3( ms->r, ms->g, ms->b );   // hmm: this is render specific, but easily ignored
    float t = 0.f ; 
    unsigned identity = 0u ; 
    unsigned boundary = 0u ; 
    setPayload( normal, t, identity, boundary, 0.f, 0.f  );
}

/**
__closesthit__ch : pass attributes from __intersection__ into setPayload
============================================================================

optixGetInstanceId 
    flat instance_idx over all transforms in the single IAS, 
    JUNO maximum ~50,000 (fits with 0xffff = 65535)

optixGetPrimitiveIndex
    local index of AABB within the GAS, 
    instanced solids adds little to the number of AABB, 
    most come from unfortunate repeated usage of prims in the non-instanced global
    GAS with repeatIdx 0 (JUNO up to ~4000)

optixGetRayTmax
    In intersection and CH returns the current smallest reported hitT or the tmax passed into rtTrace 
    if no hit has been reported


**/

extern "C" __global__ void __closesthit__ch()
{
    const float3 local_normal =    // geometry object frame normal at intersection point 
        make_float3(
                uint_as_float( optixGetAttribute_0() ),
                uint_as_float( optixGetAttribute_1() ),
                uint_as_float( optixGetAttribute_2() )
                );
   

    const float t = uint_as_float( optixGetAttribute_3() ) ;  
    unsigned boundary = optixGetAttribute_4() ; 

    const float3 local_point =    // geometry object frame normal at intersection point 
        make_float3(
                uint_as_float( optixGetAttribute_5() ),
                uint_as_float( optixGetAttribute_6() ),
                uint_as_float( optixGetAttribute_7() )
                );

    const float spare1 = optixGetRayTmax() ; 
    const float spare2 = local_point.x ; 

    //unsigned instance_index = optixGetInstanceIndex() ;  0-based index within IAS
    unsigned instance_id = optixGetInstanceId() ;  // user supplied instanceId, see IAS_Builder::Build and InstanceId.h 
    unsigned prim_idx = optixGetPrimitiveIndex() ;  // see GAS_Builder::MakeCustomPrimitivesBI_11N  (1+index-of-CSGPrim within CSGSolid/GAS)
    unsigned identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff ) ; 

    float3 normal = optixTransformNormalFromObjectToWorldSpace( local_normal ) ;  

    setPayload( normal, t, identity, boundary, spare1, spare2 );  // communicate to raygen 
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
    //int numNode = hg->numNode ;        // equivalent to CSGPrim, as same info : specify complete binary tree sequence of CSGNode 
    int nodeOffset = hg->nodeOffset ; 

    const CSGNode* node = params.node + nodeOffset ;  // root of tree
    const float4* plan = params.plan ;  
    const qat4*   itra = params.itra ;  

    const float  t_min = optixGetRayTmin() ; 
    const float3 ray_origin = optixGetObjectRayOrigin();
    const float3 ray_direction = optixGetObjectRayDirection();

    float4 isect ; // .xyz normal .w distance 
    if(intersect_prim(isect, node, plan, itra, t_min , ray_origin, ray_direction ))  
    {
        const unsigned hitKind = 0u ;   // only 8bit : could use to customize how attributes interpreted
        unsigned a0, a1, a2, a3, a4, a5, a6, a7 ;      
        const unsigned boundary = node->boundary() ;  // all nodes of tree have same boundary 

        float3 local_point = ray_origin + isect.w*ray_direction ;        

        a0 = float_as_uint( isect.x );  // isect.xyz is object frame normal of geometry at intersection point 
        a1 = float_as_uint( isect.y );
        a2 = float_as_uint( isect.z );
        a3 = float_as_uint( isect.w ) ; // perhaps no need to pass the "t", should be standard access to "t"
        a4 = boundary ; 
        a5 = float_as_uint( local_point.x ); 
        a6 = float_as_uint( local_point.y ); 
        a7 = float_as_uint( local_point.z ); 

        optixReportIntersection( isect.w, hitKind, a0, a1, a2, a3, a4, a5, a6, a7 );   
        // IS:optixReportIntersection writes the attributes that can be read in CH and AH programs 
        // max 8 attribute registers, see PIP::PIP, communicate to __closesthit__ch 
    }
}
// story begins with intersection
