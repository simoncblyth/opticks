/**
cxr_overview.sh render times with JUNO 
=======================================

JUNO (trunk:Dec, 2021)
   without   0.0054 

JUNO (trunk:Mar 2, 2022)

   without  0.0126
   WITH_PRD 0.0143
   without  0.0126
   without  0.0125  
   WITH_PRD 0.0143   (here and above WITH_PRD used attribs and payload values at 8 without reason)
   WITH_PRD 0.0125   (now with attribs and payload values reduced to 2)
   WITH_PRD 0.0124  

   WITH_PRD not-WITH_CONTIGUOUS 0.0123

**/

#include <optix.h>

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"

// simulation 
#include <curand_kernel.h>
#include "qsim.h"
#include "qevent.h"
#include "qprd.h"

#include "csg_intersect_leaf.h"
#include "csg_intersect_node.h"
#include "csg_intersect_tree.h"

#include "Binding.h"
#include "Params.h"

#ifdef WITH_PRD
#include "Pointer.h"
#endif

extern "C" { __constant__ Params params ;  }

/**
trace : pure function, with no use of params, everything via args
-------------------------------------------------------------------

Outcome of trace is to populate *prd* by payload and attribute passing.
When WITH_PRD macro is defined only 2 32-bit payload values are used to 
pass the 64-bit  pointer, otherwise more payload and attributes values 
are used to pass the contents IS->CH->RG. 

See __closesthit__ch to see where the payload p0-p5 comes from.
**/

static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        qprd*                  prd
        )   
{
    const float rayTime = 0.0f ; 
    OptixVisibilityMask visibilityMask = 1u  ; 
    //OptixRayFlags rayFlags = OPTIX_RAY_FLAG_NONE ; 
    OptixRayFlags rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT ; 
    const unsigned SBToffset = 0u ; 
    const unsigned SBTstride = 1u ; 
    const unsigned missSBTIndex = 0u ; 
#ifdef WITH_PRD
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
#else
    uint32_t p0, p1, p2, p3, p4, p5 ; 
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
            p0, p1, p2, p3, p4, p5
            );
    // unclear where the uint_as_float CUDA device function is defined, seems from optix7.h but cannot locate 
    prd->normal.x = uint_as_float( p0 );
    prd->normal.y = uint_as_float( p1 );
    prd->normal.z = uint_as_float( p2 );
    prd->t        = uint_as_float( p3 ); 
    prd->identity = p4 ; 
    prd->boundary = p5 ;  
#endif
}


__forceinline__ __device__ uchar4 make_color( const float3& normal, unsigned identity, unsigned boundary )  // pure 
{
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

static __forceinline__ __device__ void render( const uint3& idx, const uint3& dim, qprd* prd )
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

    float3 position = origin + direction*prd->t ; 
    float3 diddled_normal = normalize(prd->normal)*0.5f + 0.5f ; // diddling lightens the render, with mid-grey "pedestal" 
    unsigned index = idx.y * params.width + idx.x ;

    params.pixels[index] = make_color( diddled_normal, prd->identity, prd->boundary ); 
    params.isect[index]  = make_float4( position.x, position.y, position.z, uint_as_float(prd->identity)) ; 
}
 
/**
simulate : uses params for input: gensteps, seeds and output photons 
----------------------------------------------------------------------

Contrast with the monolithic old way with OptiXRap/cu/generate.cu:generate 

This method aims to get as much as possible of its functionality from 
separately implemented and tested headers. 

The big thing that CSGOptiX provides is geometry intersection, only that must be here. 
Everything else should be implemented and tested elsewhere, mostly in QUDARap headers.

Hence this "simulate" needs to act as a coordinator. 
Params take central role in enabling this:


Params
~~~~~~~

* CPU side params including qsim.h qevent.h pointers instanciated in CSGOptiX::CSGOptiX 
  and populated by CSGOptiX::init methods before being uploaded by CSGOptiX::prepareParam 


**/

static __forceinline__ __device__ void simulate( const uint3& idx, const uint3& dim, qprd* prd )
{
    qevent* evt      = params.evt ; 
    if (idx.x >= evt->num_photon) return;

    unsigned photon_id = idx.x ; 
    unsigned genstep_id = evt->seed[photon_id] ; 
    const quad6& gs     = evt->genstep[genstep_id] ; 
     
    qsim<float>* sim = params.sim ; 
    curandState rng = sim->rngstate[photon_id] ;    // TODO: skipahead using an event_id 

    quad4 p ;   
    qstate s ; 

    sim->generate_photon(p, rng, gs, photon_id, genstep_id );  

    float3 origin    = make_float3( p.q0.f.x, p.q0.f.y, p.q0.f.z ) ; 
    float3 direction = make_float3( p.q1.f.x, p.q1.f.y, p.q1.f.z ) ; 

    trace( 
        params.handle,
        origin,
        direction,
        params.tmin,
        params.tmax,
        prd
    );

    float cosTheta = dot(prd->normal, direction) ; 
    const float& wavelength = p.q2.f.w ; 
    sim->fill_state(s, prd->boundary, wavelength, cosTheta ); 

    // transform (x,z) intersect position into pixel coordinates (ix,iz)
    float3 ipos = origin + prd->t*direction ; 

    // aim to match the CPU side test isect "photons" from CSG/CSGQuery.cc/CSGQuery::intersect_elaborate

    p.q0.f.x = prd->normal.x ; 
    p.q0.f.y = prd->normal.y ; 
    p.q0.f.z = prd->normal.z ; 
    p.q0.f.w = prd->t  ;  

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
    p.q3.u.w = prd->identity ; 

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

    qprd prd ; 
    prd.normal = make_float3(0.5f, 0.5f, 0.5f); 
    prd.t        = 0.f ; 
    prd.identity = 0u ; 
    prd.boundary = 0u ; 

    switch( params.raygenmode )
    {
        case 0: render(   idx, dim, &prd ) ; break ;  
        case 1: simulate( idx, dim, &prd ) ; break ;  
    }
} 


#ifdef WITH_PRD
#else

/**
*setPayload* is used from __closesthit__ and __miss__ providing communication to __raygen__ optixTrace call
**/
static __forceinline__ __device__ void setPayload( float3 normal, float t, unsigned identity, unsigned boundary   ) // pure? 
{
    optixSetPayload_0( float_as_uint( normal.x ) );
    optixSetPayload_1( float_as_uint( normal.y ) );
    optixSetPayload_2( float_as_uint( normal.z ) );
    optixSetPayload_3( float_as_uint( t ) );
    optixSetPayload_4( identity );
    optixSetPayload_5( boundary );
    // maximum of 6 payload values configured in PIP::PIP 
    // NB : payload is distinct from attributes
}

#endif

extern "C" __global__ void __miss__ms()
{
    MissData* ms  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
#ifdef WITH_PRD
    qprd* prd = getPRD<qprd>(); 
    prd->normal.x =  ms->r ; 
    prd->normal.y =  ms->g ; 
    prd->normal.z =  ms->b ;
    prd->t = 0.f ; 
    prd->identity = 0u ; 
    prd->boundary = 0u ; 
#else
    float3 normal = make_float3( ms->r, ms->g, ms->b );   // hmm: this is render specific, but easily ignored
    float t = 0.f ; 
    unsigned identity = 0u ; 
    unsigned boundary = 0u ; 
    setPayload( normal, t, identity, boundary );  // communicate ms->rg
#endif
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
#ifdef WITH_PRD
    qprd* prd = getPRD<qprd>(); 
    unsigned instance_id = optixGetInstanceId() ;  // user supplied instanceId, see IAS_Builder::Build and InstanceId.h 
    unsigned prim_idx = optixGetPrimitiveIndex() ;  // GAS_Builder::MakeCustomPrimitivesBI_11N  (1+index-of-CSGPrim within CSGSolid/GAS)
    unsigned identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff ) ; 

    prd->identity = identity ;   
    prd->normal = optixTransformNormalFromObjectToWorldSpace( prd->normal ) ;  
    //prd->boundary is set in intersect 
#else
    const float3 local_normal =    // geometry object frame normal at intersection point 
        make_float3(
                uint_as_float( optixGetAttribute_0() ),
                uint_as_float( optixGetAttribute_1() ),
                uint_as_float( optixGetAttribute_2() )
                );

    const float t = uint_as_float(  optixGetAttribute_3() ) ;  
    unsigned boundary = optixGetAttribute_4() ; 

    //unsigned instance_index = optixGetInstanceIndex() ;  0-based index within IAS
    unsigned instance_id = optixGetInstanceId() ;  // user supplied instanceId, see IAS_Builder::Build and InstanceId.h 
    unsigned prim_idx = optixGetPrimitiveIndex() ;  // see GAS_Builder::MakeCustomPrimitivesBI_11N  (1+index-of-CSGPrim within CSGSolid/GAS)
    unsigned identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff ) ; 

    float3 normal = optixTransformNormalFromObjectToWorldSpace( local_normal ) ;  

    setPayload( normal, t, identity, boundary);  // communicate from ch->rg
#endif
}

/**
__intersection__is
----------------------

HitGroupData provides the numNode and nodeOffset of the intersected CSGPrim.
Which Prim gets intersected relies on the CSGPrim::setSbtIndexOffset

Note that optixReportIntersection returns a bool, but that is 
only relevant when using anyHit as it provides a way to ignore hits.
But Opticks does not used any anyHit so the returned bool should 
always be true. 

The attributes passed into optixReportIntersection are 
available within the CH (and AH) programs. 

**/

extern "C" __global__ void __intersection__is()
{
    HitGroupData* hg  = (HitGroupData*)optixGetSbtDataPointer();  
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
        const unsigned hitKind = 0u ;            // only 8bit : could use to customize how attributes interpreted
        const unsigned boundary = node->boundary() ;  // all nodes of tree have same boundary 
#ifdef WITH_PRD
        if(optixReportIntersection( isect.w, hitKind))
        {
            qprd* prd = getPRD<qprd>(); 
            prd->normal.x = isect.x ;  
            prd->normal.y = isect.y ;  
            prd->normal.z = isect.z ;
            prd->t        = isect.w ;   
            prd->boundary = boundary ; 
        }   
#else
        unsigned a0, a1, a2, a3, a4  ;      
        a0 = float_as_uint( isect.x );  // isect.xyz is object frame normal of geometry at intersection point 
        a1 = float_as_uint( isect.y );
        a2 = float_as_uint( isect.z );
        a3 = float_as_uint( isect.w ) ; // perhaps no need to pass the "t", should be standard access to "t"
        a4 = boundary ; 
        optixReportIntersection( isect.w, hitKind, a0, a1, a2, a3, a4 );   
#endif
        // IS:optixReportIntersection writes the attributes that can be read in CH and AH programs 
        // max 8 attribute registers, see PIP::PIP, communicate to __closesthit__ch 
    }
}
// story begins with intersection
