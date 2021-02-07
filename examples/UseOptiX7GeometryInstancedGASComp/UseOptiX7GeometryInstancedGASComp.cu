//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>

#include "Binding.h"
#include "sutil_vec_math.h"

extern "C" {
__constant__ Params params;
}

/**

UseOptiX7GeometryInstancedGAS.cu

700p43

    (hitgroup) 
    sbt-index = sbt-instance-offset + (sbt-GAS-index * sbt-stride-from-trace-call) + sbt-offset-from-trace-call

    sbt-GAS-index : order of GAS creation 

**/

static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        float3*                prd, 
        unsigned*              iidx  
        )
{
    uint32_t p0, p1, p2, p3;
    p0 = float_as_int( prd->x );
    p1 = float_as_int( prd->y );
    p2 = float_as_int( prd->z );
    p3 = *iidx ;

    // think that these are to handle multiple ray types 
    // eg with 2 ray types would have stride 2 and offsets 0 and 1 
    unsigned SBToffset = 0u ; 
    unsigned SBTstride = 0u ; 

    unsigned missSBTIndex = 0u ; 

    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            SBToffset,
            SBTstride,
            missSBTIndex,
            p0, p1, p2, p3 );
    prd->x = int_as_float( p0 );
    prd->y = int_as_float( p1 );
    prd->z = int_as_float( p2 );
    *iidx = p3 ; 
}


static __forceinline__ __device__ void setPayload( float3 p, unsigned instance_idx)
{
    optixSetPayload_0( float_as_int( p.x ) );
    optixSetPayload_1( float_as_int( p.y ) );
    optixSetPayload_2( float_as_int( p.z ) );
    optixSetPayload_3( instance_idx );
}

/**
static __forceinline__ __device__ void getPayload(float3& p, unsigned& instance_idx)
{
    p.x = int_as_float( optixGetPayload_0() );
    p.y = int_as_float( optixGetPayload_1() );
    p.z = int_as_float( optixGetPayload_2() );
    instance_idx = optixGetPayload_3() ; 
}
**/

__forceinline__ __device__ uchar4 make_color( const float3&  c, unsigned iidx )
{
    //float scale = iidx % 2u == 0u ? 0.5f : 1.f ; 
    float scale = 1.f ; 
    return make_uchar4(
            static_cast<uint8_t>( clamp( c.x, 0.0f, 1.0f ) *255.0f )*scale ,
            static_cast<uint8_t>( clamp( c.y, 0.0f, 1.0f ) *255.0f )*scale ,
            static_cast<uint8_t>( clamp( c.z, 0.0f, 1.0f ) *255.0f )*scale ,
            255u
            );
}

/**
__raygen__rg
----------------

Q: Why get "uniform" camera data form from sbt ? When 
   changing it requires rebuilding sbt as opposed 
   to getting it from __constant__ memory params ?

**/

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();
    const float3      U      = rtData->camera_u;
    const float3      V      = rtData->camera_v;
    const float3      W      = rtData->camera_w;

     const float   tmin = rtData->tmin ; 
     const float   tmax = rtData->tmax ; 

    const float2      d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
            ) - 1.0f;

    const float3 origin      = rtData->cam_eye;
    const float3 direction   = normalize( d.x * U + d.y * V + W );
    float3       payload_rgb = make_float3( 0.5f, 0.5f, 0.5f );
    unsigned instance_idx = 0u ; 
    trace( params.handle,
            origin,
            direction,
            tmin,
            tmax,
            &payload_rgb, 
            &instance_idx );

    params.image[idx.y * params.image_width + idx.x] = make_color( payload_rgb, instance_idx );
}

extern "C" __global__ void __miss__ms()
{
    MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    float3    p = make_float3( 0.f, 0.f, 0.f); 
    unsigned instance_idx = 0 ; 
    //getPayload(&p, &instance_idx);

    setPayload( make_float3( rt_data->r, rt_data->g, rt_data->b ), instance_idx );
}

/**


How could an __intersection__analytic_csg_list work ?

* can see how to branch to different shapes based on content of Sbt 
  but how do you associate the sbt records to custom geomtry bb ? 

**/

extern "C" __global__ void __intersection__is()
{
    HitGroupData* hg_data  = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
    const float3 orig = optixGetObjectRayOrigin();
    const float3 dir  = optixGetObjectRayDirection();

    const float3 center = {0.f, 0.f, 0.f};
    const float  radius = hg_data->radius;
    const float3 O      = orig - center;
    const float  l      = 1.f / length( dir );
    const float3 D      = dir * l;

    const float b    = dot( O, D );
    const float c    = dot( O, O ) - radius * radius;
    const float disc = b * b - c;
    if( disc > 0.0f )
    {
        const float sdisc = sqrtf( disc );
        const float root1 = ( -b - sdisc );

        const float        root11        = 0.0f;
        const float3       shading_normal = ( O + ( root1 + root11 ) * D ) / radius;
        unsigned int p0, p1, p2;
        p0 = float_as_int( shading_normal.x );
        p1 = float_as_int( shading_normal.y );
        p2 = float_as_int( shading_normal.z );

        optixReportIntersection(
                root1,      // t hit
                0,          // user hit kind
                p0, p1, p2
                );
    }
}

extern "C" __global__ void __closesthit__ch()
{
    const float3 shading_normal =
        make_float3(
                int_as_float( optixGetAttribute_0() ),
                int_as_float( optixGetAttribute_1() ),
                int_as_float( optixGetAttribute_2() )
                );

    //OptixTraversableHandle gas = optixGetGASTraversableHandle(); 
    unsigned instance_idx = optixGetInstanceIndex() ;

    setPayload( normalize( optixTransformNormalFromObjectToWorldSpace( shading_normal ) ) * 0.5f + 0.5f , instance_idx );
}

