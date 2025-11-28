/**
CSGOptiX7.cu
===================

NB: ONLY CODE THAT MUST BE HERE DUE TO OPTIX DEPENDENCY SHOULD BE HERE
everything else should be located elsewhere : mostly in qudarap: sevent, qsim
or the sysrap basis types sphoton quad4 quad2 etc.. where the code is reusable
and more easily tested.

Functions
-----------

trace
    populate quad2 prd by call to optixTrace

make_color
    minimal normal "shader"

render
    raygen function : calling trace and "shading" pixels

simulate
    raygen function : qsim::generate_photon, bounce while loop, qsim::propagate

    * ifndef PRODUCTION sctx::trace sctx::point record the propagation point-by-point

simtrace
    raygen function : qsim.h generate_photon_simtrace, trace, sevent::add_simtrace

__raygen__rg
    calls one of the above raygen functions depending on params.raygenmode

setPayload
    mechanics of communication when not using WITH_PRD

__miss_ms
    default quad2 prd OR payload for rays that miss

__closesthit__ch
    populate quad2 prd OR payload for rays that intersect

__intersection__is
    converts OptiX HitGroupData into corresponding CSGNode and calls intersect_prim
    giving float4 isect: (normal_at_intersect, distance)

**/

#include <optix.h>

#include "SRG.h"
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "sphoton.h"
#include "sphotonlite.h"
#include "scerenkov.h"
#include "sstate.h"

#ifndef PRODUCTION
#include "stag.h"
#include "sseq.h"
#include "srec.h"
#endif

#include "sevent.h"
#include "sctx.h"

// simulation
#include <curand_kernel.h>

#include "qrng.h"
#include "qsim.h"

#include "csg_intersect_leaf.h"
#include "csg_intersect_node.h"
#include "csg_intersect_tree.h"

#include "Binding.h"
#include "Params.h"

#ifdef WITH_PRD
#include "scuda_pointer.h"
#include "SOPTIX_getPRD.h"
#endif

extern "C" { __constant__ Params params ;  }

/**
trace : pure function, with no use of params, everything via args
-------------------------------------------------------------------

refine:false
    does single optixTrace

refine:true
    does a second optixTrace if 99 percent of the distance returned by the
    first optixTrace exceeds the refine_distance argument.
    This attempts to improve the precision of long distance intersects
    by doing a 2nd closer intersect.


Outcome of trace is to populate *prd* by payload and attribute passing.
When WITH_PRD macro is defined only 2 32-bit payload values are used to
pass the 64-bit  pointer, otherwise more payload and attributes values
are used to pass the contents IS->CH->RG.

See __closesthit__ch to see where the payload p0-p7 comes from.
**/

template<bool refine>
static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        quad2*                 prd,
        unsigned               visibilityMask,
        float                  refine_distance
        )
{
    const float rayTime = 0.0f ;
    OptixRayFlags rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT ;   // OPTIX_RAY_FLAG_NONE
    const unsigned SBToffset = 0u ;
    const unsigned SBTstride = 1u ;
    const unsigned missSBTIndex = 0u ;
#ifdef WITH_PRD
    uint32_t p0, p1 ;
    packPointer( prd, p0, p1 ); // scuda_pointer.h : pack prd addr from RG program into two uint32_t passed as payload

    if(!refine)
    {
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
    else
    {
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

        float t_approx = 0.99f*prd->distance() ;
        // decrease to avoid imprecise first intersect from
        // leading to giving a miss or other intersect the 2nd time
        if( t_approx > refine_distance )
        {
            float3 closer_ray_origin = ray_origin + t_approx*ray_direction ;
            optixTrace(
                handle,
                closer_ray_origin,
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

            prd->distance_add( t_approx );
        }
    }

#else
    uint32_t p0, p1, p2, p3, p4, p5, p6, p7  ;
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
            p0, p1, p2, p3, p4, p5, p6, p7
            );
    // unclear where the uint_as_float CUDA device function is defined, seems CUDA intrinsic without header ?
    prd->q0.f.x = __uint_as_float( p0 );
    prd->q0.f.y = __uint_as_float( p1 );
    prd->q0.f.z = __uint_as_float( p2 );
    prd->q0.f.w = __uint_as_float( p3 );
    prd->set_identity(p4) ;
    prd->set_globalPrimIdx_boundary_(p5) ;
    prd->set_lposcost(__uint_as_float(p6)) ;   // trace.not-WITH_PRD
    prd->set_iindex(p7) ;
#endif
}

//#if !defined(PRODUCTION) && defined(WITH_RENDER)
#if defined(WITH_RENDER)

__forceinline__ __device__ uchar4 make_normal_pixel( const float3& normal, float depth )  // pure
{
    return make_uchar4(
            static_cast<uint8_t>( clamp( normal.x, 0.0f, 1.0f ) *255.0f ),
            static_cast<uint8_t>( clamp( normal.y, 0.0f, 1.0f ) *255.0f ),
            static_cast<uint8_t>( clamp( normal.z, 0.0f, 1.0f ) *255.0f ),
            static_cast<uint8_t>( clamp( depth   , 0.0f, 1.0f ) *255.0f )
            );
}

__forceinline__ __device__ uchar4 make_zdepth_pixel( float depth )  // pure
{
    return make_uchar4(
            static_cast<uint8_t>( clamp( depth   , 0.0f, 1.0f ) *255.0f ),
            static_cast<uint8_t>( clamp( depth   , 0.0f, 1.0f ) *255.0f ),
            static_cast<uint8_t>( clamp( depth   , 0.0f, 1.0f ) *255.0f ),
            static_cast<uint8_t>( clamp( depth   , 0.0f, 1.0f ) *255.0f )
            );
}



/**
render : non-pure, uses params for viewpoint inputs and pixels output
-----------------------------------------------------------------------

Bugs with normal(0.f,0.f,0.f) via normalizing yields diddled_normal(nan,nan,nan)
which make_color manages to clamp to (0,0,0) black.

**/

static __forceinline__ __device__ void render( const uint3& idx, const uint3& dim, quad2* prd )
{

#if defined(DEBUG_PIDX)
    //if(idx.x == 10 && idx.y == 10) printf("//CSGOptiX7.cu:render idx(%d,%d,%d) dim(%d,%d,%d) \n", idx.x, idx.y, idx.z, dim.x, dim.y, dim.z );
#endif

    float2 d = 2.0f * make_float2(
            static_cast<float>(idx.x)/static_cast<float>(dim.x),
            static_cast<float>(idx.y)/static_cast<float>(dim.y)
            ) - 1.0f;


    const unsigned cameratype = params.cameratype ;
    const float3 dxyUV = d.x * params.U + params.V * ( params.traceyflip ? -d.y : d.y ) ;
    const float3 origin    = cameratype == 0u ? params.eye                     : params.eye + dxyUV    ;
    const float3 direction = cameratype == 0u ? normalize( dxyUV + params.W )  : normalize( params.W ) ;
    //                           cameratype 0u:perspective,                    1u:orthographic

    trace<false>(
        params.handle,
        origin,
        direction,
        params.tmin,
        params.tmax,
        prd,
        params.vizmask,
        params.PropagateRefineDistance
    );

#if defined(DEBUG_PIDX)
    //if(idx.x == 10 && idx.y == 10) printf("//CSGOptiX7.cu:render prd.distance(%7.3f)  prd.lposcost(%7.3f)  \n", prd->distance(), prd->lposcost()  );
#endif





    const float3* normal = prd->normal();

#if defined(DEBUG_PIDX)
    //if(idx.x == 10 && idx.y == 10) printf("//CSGOptiX7.cu:render normal(%7.3f,%7.3f,%7.3f)  \n", normal->x, normal->y, normal->z );
#endif

    float3 diddled_normal = normalize(*normal)*0.5f + 0.5f ; // diddling lightens the render, with mid-grey "pedestal"

    float eye_z = -prd->distance()*dot(params.WNORM, direction) ;
    const float& A = params.ZPROJ.z ;
    const float& B = params.ZPROJ.w ;
    float zdepth = cameratype == 0u ? -(A + B/eye_z) : A*eye_z + B  ;  // cf SGLM::zdepth1

    if( prd->is_boundary_miss() ) zdepth = 0.999f ;
    // setting miss zdepth to 1.f give black miss pixels, 0.999f gives expected mid-grey from normal of (0.f,0.f,0.f)
    // previously with zdepth of zero for miss pixels found that OpenGL record rendering did not
    // appear infront of the grey miss pixels : because they were behind them (zdepth > 0.f ) presumably

    unsigned index = idx.y * params.width + idx.x ;

    if(params.pixels)
    {
#if defined(DEBUG_PIDX)
        //if(idx.x == 10 && idx.y == 10) printf("//CSGOptiX7.cu:render/params.pixels diddled_normal(%7.3f,%7.3f,%7.3f)  \n", diddled_normal.x, diddled_normal.y, diddled_normal.z );
#endif
        params.pixels[index] = params.rendertype == 0 ? make_normal_pixel( diddled_normal, zdepth ) : make_zdepth_pixel( zdepth ) ;
    }
    if(params.isect)
    {
        float3 position = origin + direction*prd->distance() ;
        params.isect[index]  = make_float4( position.x, position.y, position.z, __uint_as_float(prd->identity())) ;
    }


}
#endif


#if defined(WITH_SIMULATE)

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

* CPU side params including qsim.h sevent.h pointers instanciated in CSGOptiX::CSGOptiX
  and populated by CSGOptiX::init methods before being uploaded by CSGOptiX::prepareParam

COMPARE WITH qsim::mock_propagate



Is cycling of the launch_idx.x (unsigned)idx possible here ? NOT YET
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* 0xffffffff/1e9 = 4.29 billion photons in one launch would need VRAM of 275 GB::

    In [102]: (0xffffffff*4*4*4)/1e9
    Out[102]: 274.87790688   # 275 GB of photons in one launch ?


* suggests should limit max_slot to 0xffffffff (4.29 billion, on top of VRAM requirements)
* current heuristic limit of 250M photons when limited by 32 GB VRAM

* NOTE THE ABSOLUTE photon_idx IS LIKELY BEING CYCLED HOWEVER


tmin/tmin0 avoiding light leakage "tunnelling"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ray trace tmin is used to prevent rays with origin "on" a surface (ie very close to it)
from intersecting with the surface when trying to escape from it.
For example on reflecting from a surface multiple intersects can be avoided
with a suitably chosed tmin. Using the same tmin for all steps of the photon is simple
but can have undesirable consequences. For example when generation by scintillation/cerenkov/torch
or scatter/reemission happens within tmin of a surface the photon might "tunnel"
through the surface. Globally reducing tmin would reduce the probability of tunneling but
would cause repeated intersection issues for rays starting "on" surfaces.

To avoid this dilemma a second tmin0 and the photon step flags that cause
tmin0 to be used for the next ray trace can be configured.
As shown in the below table the default photon step flags that cause tmin0
to be used for the next ray trace are TO,CK,SI,SC,RE which
are all things (generation, scattering, reemission) that mostly happen
in the bulk away from boundaries where there is less concern regarding
self intersection. As self-intersection is not so much of a concern tmin0
might be set to zero. However, there could then be a problem with self intersection
in cases where these things happen within close to surfaces.

+----------------------------------+-----------------------------------------+-------------------+
|  envvar                          |  method                                 | default           |
+==================================+=========================================+===================+
| OPTICKS_PROPAGATE_EPSILON        |  SEventConfig::PropagateEpsilon()       |  0.05f            |
+----------------------------------+-----------------------------------------+-------------------+
| OPTICKS_PROPAGATE_EPSILON0       |  SEventConfig::PropagateEpsilon0()      |  0.05f            |
+----------------------------------+-----------------------------------------+-------------------+
| OPTICKS_PROPAGATE_EPSILON0_MASK  |  SEventConfig::PropagateEpsilon0Mask()  |  TO,CK,SI,SC,RE   |
+----------------------------------+-----------------------------------------+-------------------+

**/

static __forceinline__ __device__ void simulate( const uint3& launch_idx, const uint3& dim, quad2* prd )
{
    sevent* evt = params.evt ;
    if (launch_idx.x >= evt->num_seed) return;   // was evt->num_photon

    unsigned idx = launch_idx.x ;
    unsigned genstep_idx = evt->seed[idx] ;
    const quad6& gs = evt->genstep[genstep_idx] ;
    // genstep needs the raw index, from zero for each genstep slice sub-launch

    unsigned long long photon_idx = params.photon_slot_offset + idx ;
    // 2025/10/20 change from unsigned to avoid clocking photon_idx and duplicating
    //
    // rng_state access and array recording needs the absolute photon_idx
    // for multi-launch and single-launch simulation to match.
    // The offset hides the technicality of the multi-launch from output.

    qsim* sim = params.sim ;

//#define OLD_WITHOUT_SKIPAHEAD 1
#ifdef OLD_WITHOUT_SKIPAHEAD
    RNG rng = sim->rngstate[photon_idx] ;
#else
    RNG rng ;
    sim->rng->init( rng, sim->evt->index, photon_idx );
#endif

    sctx ctx = {} ;
    ctx.evt = evt ;   // sevent.h
    ctx.prd = prd ;   // squad.h quad2

    ctx.idx = idx ;
    ctx.pidx = photon_idx ;

#if !defined(PRODUCTION) && defined(DEBUG_PIDX)
    ctx.pidx_debug = sim->base->pidx == photon_idx ;
#endif

    sim->generate_photon(ctx.p, rng, gs, photon_idx, genstep_idx );

    int command = START ;
    int bounce = 0 ;
#ifndef PRODUCTION
    ctx.point(bounce);
#endif
    while( bounce < evt->max_bounce && ctx.p.time < params.max_time )
    {
        float tmin = ( ctx.p.orient_boundary_flag & params.PropagateEpsilon0Mask ) ? params.tmin0 : params.tmin ;

        // intersect query filling (quad2)prd
        switch(params.PropagateRefine)
        {
            case 0u: trace<false>( params.handle, ctx.p.pos, ctx.p.mom, tmin, params.tmax, prd, params.vizmask, params.PropagateRefineDistance );  break ;
            case 1u: trace<true>(  params.handle, ctx.p.pos, ctx.p.mom, tmin, params.tmax, prd, params.vizmask, params.PropagateRefineDistance );  break ;
        }

        if( prd->boundary() == 0xffffu ) break ; // SHOULD ONLY HAPPEN FOR PHOTONS STARTING OUTSIDE WORLD
        // propagate can do nothing meaningful without a boundary

        // HMM: normalize here or within CSG ? Actually only needed for
        // geometry with active scaling, such as ellipsoid.
        // TODO: move this so its only done when needed
        //     ~/o/notes/issues/CSGOptiX_simulate_avoid_normalizing_every_normal.rst
        //

        float3* normal = prd->normal();
        *normal = normalize(*normal);

#ifndef PRODUCTION
        ctx.trace(bounce);
#endif
        command = sim->propagate(bounce, rng, ctx);
        bounce++;
#ifndef PRODUCTION
        ctx.point(bounce) ;
#endif
        if(command == BREAK) break ;
    }
#ifndef PRODUCTION
    ctx.end();  // write seq, tag, flat
#endif


    if( evt->photon )
    {
        evt->photon[idx] = ctx.p ;  // *idx* (not *photon_idx*) as needs to go from zero for photons from a slice of genstep array
    }

    if( evt->photonlite )
    {
        sphotonlite l ;
        l.init( ctx.p.identity, ctx.p.time, ctx.p.flagmask );
        l.set_lpos(prd->lposcost(), prd->lposfphi() );
        evt->photonlite[idx] = l ;  // *idx* (not *photon_idx*) as needs to go from zero for photons from a slice of genstep array
    }



}

#endif


//#if !defined(PRODUCTION) && defined(WITH_SIMTRACE)
#if defined(WITH_SIMTRACE)

/**
simtrace
----------

Used for making 2D cross section views of geometry intersects

Note how seeding is still needed here despite the highly artificial
nature of the center-extent grid of gensteps as the threads of the launch
still needs to access different gensteps across the grid.

TODO: Compose frames of pixels, isect and "fphoton" within the cegs window
using the positions of the intersect "photons".
Note that multiple threads may be writing to the same pixel
that is apparently not a problem, just which does it is uncontrolled.

unsigned index = iz * params.width + ix ;
if( index > 0 )
{
    params.pixels[index] = make_uchar4( 255u, 0u, 0u, 255u) ;
    params.isect[index] = make_float4( ipos.x, ipos.y, ipos.z, uint_as_float(identity)) ;
    params.fphoton[index] = p ;
}
**/


static __forceinline__ __device__ void simtrace( const uint3& launch_idx, const uint3& dim, quad2* prd )
{
    unsigned idx = launch_idx.x ;
    sevent* evt  = params.evt ;
    if (idx >= evt->num_simtrace) return;    // num_slot for multi launch simtrace ?

    unsigned genstep_idx = evt->seed[idx] ;
    unsigned photon_idx  = params.photon_slot_offset + idx ;
    // photon_idx same as idx for first launch, offset beyond first for multi-launch

#if defined(DEBUG_PIDX)
    if(photon_idx == 0) printf("//CSGOptiX7.cu : simtrace idx %d photon_idx %d  genstep_idx %d evt->num_simtrace %ld \n", idx, photon_idx, genstep_idx, evt->num_simtrace );
#endif

    const quad6& gs = evt->genstep[genstep_idx] ;

    qsim* sim = params.sim ;
    RNG rng ;
    sim->rng->init(rng, 0, photon_idx) ;

    quad4 p ;
    sim->generate_photon_simtrace(p, rng, gs, photon_idx, genstep_idx );


    // HUH: this is not the layout of sevent::add_simtrace
    const float3& pos = (const float3&)p.q0.f  ;
    const float3& mom = (const float3&)p.q1.f ;


#if defined(DEBUG_PIDX)
    if(photon_idx == 0) printf("//CSGOptiX7.cu : simtrace idx %d pos.xyz %7.3f,%7.3f,%7.3f mom.xyz %7.3f,%7.3f,%7.3f  \n", idx, pos.x, pos.y, pos.z, mom.x, mom.y, mom.z );
#endif

    switch(params.PropagateRefine)
    {
        case 0u: trace<false>( params.handle, pos, mom, params.tmin, params.tmax, prd, params.vizmask, params.PropagateRefineDistance );  break ;
        case 1u: trace<true>(  params.handle, pos, mom, params.tmin, params.tmax, prd, params.vizmask, params.PropagateRefineDistance );  break ;
    }

    evt->add_simtrace( idx, p, prd, params.tmin );  // sevent
    // not photon_idx, needs to go from zero for photons from a slice of genstep array
}
#endif

/**
for angular efficiency need intersection point in object frame to get the angles
**/

extern "C" __global__ void __raygen__rg_dummy()
{
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    //bool midpix = idx.x == dim.x/2 && idx.y == dim.y/2 && idx.z == dim.z/2 ;
    //if(midpix) printf("//__raygen_rg.midpix params.raygenmode %d  \n", params.raygenmode);

    quad2 prd ;
    prd.zero();


    switch( params.raygenmode )
    {

#ifdef WITH_SIMULATE
        case SRG_SIMULATE:  simulate( idx, dim, &prd ) ; break ;
#endif
#ifdef WITH_RENDER
        case SRG_RENDER:    render(   idx, dim, &prd ) ; break ;
#endif
#ifdef WITH_SIMTRACE
        case SRG_SIMTRACE:  simtrace( idx, dim, &prd ) ; break ;
#endif
    }

}


#ifdef WITH_PRD
#else
/**
*setPayload* is used from __closesthit__ and __miss__ providing communication to __raygen__ optixTrace call

NB THESE QWN NEED NOT BE THE SAME AS THE ATTRIB USED TO COMMUNICATE BETWEEN __intersection__is and __closesthit__ch

**/
static __forceinline__ __device__ void setPayload(
     float    normal_x,        float normal_y,                  float normal_z, float distance,
     unsigned iindex_identity, unsigned globalPrimIdx_boundary, float lposcost, float lposfphi )
{
    optixSetPayload_0( __float_as_uint( normal_x ) );
    optixSetPayload_1( __float_as_uint( normal_y ) );
    optixSetPayload_2( __float_as_uint( normal_z ) );
    optixSetPayload_3( __float_as_uint( distance ) );
    optixSetPayload_4( iindex_identity );
    optixSetPayload_5( globalPrimIdx_boundary );
    optixSetPayload_6( lposcost );
    optixSetPayload_7( lposfphi );

    // num_payload_values PIP::PIP must match the payload slots used up to maximum of 8
    // NB : payload is distinct from attributes
}
#endif

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
    MissData* ms  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    const unsigned ii_id = 0xffffffffu ;
    const unsigned gp_bd = 0xffffffffu ;
    const float lposcost = 0.f ;
    const float lposfphi = 0.f ;

#ifdef WITH_PRD
    quad2* prd = SOPTIX_getPRD<quad2>();

    prd->q0.f.x = ms->r ;
    prd->q0.f.y = ms->g ;
    prd->q0.f.z = ms->b ;
    prd->q0.f.w = 1.f ;     // attempt to allow rendering infront of miss pixels, but dont work

    prd->q1.u.x = 0u ;
    prd->q1.u.y = 0u ;
    prd->q1.u.z = 0u ;
    prd->q1.u.w = 0u ;

    prd->set_iindex_identity_(ii_id);
    prd->set_globalPrimIdx_boundary_(gp_bd);
    prd->set_lpos(lposcost, lposfphi);   // __miss__ms.WITH_PRD
#else
    setPayload( ms->r, ms->g, ms->b, 0.f, ii_id, gp_bd, lposcost, lposfphi );  // communicate from ms->rg
#endif
}

/**
__closesthit__ch : pass attributes from __intersection__ into setPayload
============================================================================

optixGetInstanceIndex (aka iindex)
    0-based index within IAS

optixGetInstanceId (aka identity)
    user supplied instanceId,
    see IAS_Builder::Build, sysrap/sqat4.h sqat4::get_IAS_OptixInstance_instanceId
    from July 2023: carries sensor_identifier+1 as needed for QPMT

optixGetPrimitiveIndex (aka prim_idx)
    (not currently propagated)
    local index of AABB within the GAS,
    see GAS_Builder::MakeCustomPrimitivesBI_11N  (1+index-of-CSGPrim within CSGSolid/GAS).
    Note that instanced solids adds little to the number of AABB,
    most come from unfortunate repeated usage of prims in the non-instanced global
    GAS with repeatIdx 0 (JUNO up to ~4000)

optixGetRayTmax
    In intersection and CH returns the current smallest reported hitT or the tmax passed into rtTrace
    if no hit has been reported


optixGetPrimitiveType
    OPTIX_PRIMITIVE_TYPE_CUSTOM   = 0x2500    ## 9472 : GET THIS
    OPTIX_PRIMITIVE_TYPE_TRIANGLE = 0x2531    ## 9521 : HUH:GETTING ZERO WHEN EXPECT THIS ?


ana:CH
   intersect IS program populates most of the prd per-ray-data struct
   including the trace distance and local normal at intersect
   this CH program just adds instance info and transforms the normal
   from object to world space

tri:CH
   builtin triangles have no user defined intersect program, so this tri:CH
   program must populate everything that the ana:IS and ana:CH does

tri:boundary


ana:normals
   Calculation done by each shape implementation, no other choice ?

tri:normals
   Possibilities:

   1. normal from cross product of vertex positions,
      GPU float precision calc

   2. normal from barycentric weighted vertex normals
      (probably best for sphere/torus or similar with small triangles)

   3. pick normal of one of the vertices,
      profits from double precision vertex normal calculated
      ahead of time
      (probably best for cube or similar with large triangles)

   Which is best depends on the shape and how the input
   vertex normals are calculated.

**/

extern "C" __global__ void __closesthit__ch()
{
    unsigned iindex = optixGetInstanceIndex() ;
    unsigned identity = optixGetInstanceId() ;
    unsigned iindex_identity = (( iindex & 0xffffu ) << 16 ) | ( identity & 0xffffu ) ;

    OptixPrimitiveType type = optixGetPrimitiveType(); // HUH: getting type 0, when expect OPTIX_PRIMITIVE_TYPE_TRIANGLE
    const HitGroupData* hg = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );

#if defined(DEBUG_PIDX)
    //const uint3 idx = optixGetLaunchIndex();
    //if(idx.x == 10 && idx.y == 10) printf("//__closesthit__ch idx(%u,%u,%u) type %d \n", idx.x, idx.y, idx.z, type);
    //if(identity == 52264 || identity == 52265 || identity == 52266) printf("//__closesthit__ch iindex %u type %d identity %d \n", iindex, type, identity );
#endif

    if(type == OPTIX_PRIMITIVE_TYPE_TRIANGLE || type == 0)  // WHY GETTING ZERO HERE ?
    {
        const TriMesh& mesh = hg->mesh ;
        unsigned globalPrimIdx_boundary = (( mesh.globalPrimIdx & 0xffffu ) << 16 ) | ( mesh.boundary & 0xffffu ) ;
        const unsigned prim_idx = optixGetPrimitiveIndex();
        const float2   barys    = optixGetTriangleBarycentrics();

        uint3 tri = mesh.indice[ prim_idx ];
        const float3 P0 = mesh.vertex[ tri.x ];
        const float3 P1 = mesh.vertex[ tri.y ];
        const float3 P2 = mesh.vertex[ tri.z ];
        const float3 P = ( 1.0f-barys.x-barys.y)*P0 + barys.x*P1 + barys.y*P2;

        //const float3 N0 = mesh.normal[ tri.x ];
        //const float3 N1 = mesh.normal[ tri.y ];
        //const float3 N2 = mesh.normal[ tri.z ];
        //const float3 Ng = ( 1.0f-barys.x-barys.y)*N0 + barys.x*N1 + barys.y*N2; // guesss
        // local normal from  bary-weighted vertex normals

        const float3 Ng = cross( P1-P0, P2-P0 );
        // local normal from cross product of vectors between vertices : HMM is winding order correct : TODO: check sense of normal

        const float3 N = normalize( optixTransformNormalFromObjectToWorldSpace( Ng ) );

        float t = optixGetRayTmax() ;

        // cannot get Object frame ray_origin/direction in CH (only IS,AH)
        //const float3 ray_origin = optixGetObjectRayOrigin();
        //const float3 ray_direction = optixGetObjectRayDirection();
        //const float3 lpos = ray_origin + t*ray_direction  ;
        // HMM: could use P to give the local position ?

        float lposcost = normalize_cost(P); // scuda.h  "cosTheta" z/len of local frame position
        float lposfphi = normalize_fphi(P);


#ifdef WITH_PRD
        quad2* prd = SOPTIX_getPRD<quad2>();

        prd->q0.f.x = N.x ;
        prd->q0.f.y = N.y ;
        prd->q0.f.z = N.z ;
        prd->q0.f.w = t ;

        prd->set_iindex_identity_( iindex_identity ) ;
        prd->set_globalPrimIdx_boundary_(  globalPrimIdx_boundary ) ;
        prd->set_lpos(lposcost, lposfphi);   // __closesthit__ch.WITH_PRD.TRIANGLE

#else
        setPayload( N.x, N.y, N.z, t, iindex_identity, globalPrimIdx_boundary, lposcost, lposfphi );  // communicate from ch->rg
#endif
    }
    else if(type == OPTIX_PRIMITIVE_TYPE_CUSTOM)
    {
        //const CustomPrim& cpr = hg->prim ;
#ifdef WITH_PRD
        quad2* prd = SOPTIX_getPRD<quad2>();

        prd->set_iindex_identity_( iindex_identity ) ;

        float3* normal = prd->normal();
        *normal = optixTransformNormalFromObjectToWorldSpace( *normal ) ;
#else

        // NB SEE
        const float3 local_normal =    // geometry object frame normal at intersection point
            make_float3(
                    __uint_as_float( optixGetAttribute_0() ),
                    __uint_as_float( optixGetAttribute_1() ),
                    __uint_as_float( optixGetAttribute_2() )
                    );

        const float distance = __uint_as_float(  optixGetAttribute_3() ) ;
        unsigned globalPrimIdx_boundary = optixGetAttribute_4() ;
        const float lposcost = __uint_as_float( optixGetAttribute_5() ) ;
        const float lposfphi = 0.f ; // NOT IMPL WHEN NOT:WITH_PRD

        float3 normal = optixTransformNormalFromObjectToWorldSpace( local_normal ) ;

        setPayload( normal.x, normal.y, normal.z, distance, iindex_identity, globalPrimIdx_boundary, lposcost, lposfphi );  // communicate from ch->rg
                //   p0       p1        p2        p3        p4               p5                      p6        p7
#endif
    }
}

/**
__intersection__is
----------------------

HitGroupData provides the numNode and nodeOffset of the intersected CSGPrim.
Which Prim gets intersected relies on the CSGPrim::setSbtIndexOffset

Note that optixReportIntersection returns a bool, but that is
only relevant when using anyHit as it provides a way to ignore hits.
But Opticks does not use anyHit so the returned bool should
always be true.

The attributes passed into optixReportIntersection are
available within the CH (and AH) programs.

HMM: notice that HitGroupData::numNode is not used here, must be looking that up ?
COULD: reduce HitGroupData to just the nodeOffset

**/

extern "C" __global__ void __intersection__is()
{

#if defined(DEBUG_PIDXYZ)
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    bool dumpxyz = idx.x == params.pidxyz.x && idx.y == params.pidxyz.y && idx.z == params.pidxyz.z ;
    //if(dumpxyz) printf("//__intersection__is  idx(%u,%u,%u) dim(%u,%u,%u) dumpxyz:%d \n", idx.x, idx.y, idx.z, dim.x, dim.y, dim.z, dumpxyz);
#else
    bool dumpxyz = false ;
#endif


    HitGroupData* hg  = (HitGroupData*)optixGetSbtDataPointer();
    int nodeOffset = hg->prim.nodeOffset ;
    int globalPrimIdx = hg->prim.globalPrimIdx ;

    const CSGNode* node = params.node + nodeOffset ;  // root of tree
    const float4* plan = params.plan ;
    const qat4*   itra = params.itra ;

    const float  t_min = optixGetRayTmin() ;
    const float3 ray_origin = optixGetObjectRayOrigin();
    const float3 ray_direction = optixGetObjectRayDirection();

    float4 isect ; // .xyz normal .w distance
    isect.x = 0.f ;
    isect.y = 0.f ;
    isect.z = 0.f ;
    isect.w = 0.f ;

    bool valid_isect = intersect_prim(isect, node, plan, itra, t_min , ray_origin, ray_direction, dumpxyz );
    if(valid_isect)
    {
        const float3 lpos = ray_origin + isect.w*ray_direction ;
        const float lposcost = normalize_cost(lpos);  // scuda.h cosTheta z/len of local(aka Object) frame position
        const float lposfphi = normalize_fphi(lpos);

        const unsigned hitKind = 0u ;     // only up to 127:0x7f : could use to customize how attributes interpreted
        const unsigned boundary = node->boundary() ;  // all CSGNode in the tree for one CSGPrim tree have same boundary
        const unsigned globalPrimIdx_boundary = (( globalPrimIdx & 0xffffu ) << 16 ) | ( boundary & 0xffffu ) ;

#ifdef WITH_PRD
        if(optixReportIntersection( isect.w, hitKind))
        {
            quad2* prd = SOPTIX_getPRD<quad2>(); // access prd addr from RG program
            prd->q0.f = isect ;  // .w:distance and .xyz:normal which starts as the local frame one
            prd->set_globalPrimIdx_boundary_(globalPrimIdx_boundary) ;
            prd->set_lpos(lposcost, lposfphi);    // __intersection__is.WITH_PRD.CUSTOM
        }
#else
       // TODO: REMOVE NOT:WITH_PRD
        unsigned a0, a1, a2, a3, a4, a5  ; // MUST CORRESPOND TO num_attribute_values in PIP::PIP
        a0 = __float_as_uint( isect.x );     // isect.xyz is object frame normal of geometry at intersection point
        a1 = __float_as_uint( isect.y );
        a2 = __float_as_uint( isect.z );
        a3 = __float_as_uint( isect.w ) ;
        a4 = globalPrimIdx_boundary ;
        a5 = __float_as_uint( lposcost );
        optixReportIntersection( isect.w, hitKind, a0, a1, a2, a3, a4, a5 );

        // IS:optixReportIntersection writes the attributes that can be read in CH and AH programs
        // max 8 attribute registers, see PIP::PIP, communicate to __closesthit__ch
#endif

   }

#if defined(DEBUG_PIDXYZ)
    //if(dumpxyz) printf("//__intersection__is  idx(%u,%u,%u) dim(%u,%u,%u) dumpxyz:%d valid_isect:%d\n", idx.x, idx.y, idx.z, dim.x, dim.y, dim.z, dumpxyz, valid_isect);
#endif

}
// story begins with intersection
