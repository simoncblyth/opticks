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
#include "Params.h"
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
        float3*                normal, 
        float*                 t, 
        unsigned*              iidx,
        float3*                position  
        )
{
    uint32_t p0, p1, p2, p3 ;
    uint32_t p4, p5, p6, p7 ;

    p0 = float_as_int( normal->x );
    p1 = float_as_int( normal->y );
    p2 = float_as_int( normal->z );
    p3 = float_as_int( *t );

    p4 = *iidx ;
    p5 = float_as_int( position->x );
    p6 = float_as_int( position->y );
    p7 = float_as_int( position->z );
 
    unsigned SBToffset = 0u ; 
    unsigned SBTstride = 1u ; 
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
            p0, p1, p2, p3, 
            p4, p5, p6, p7
            );

    normal->x = int_as_float( p0 );
    normal->y = int_as_float( p1 );
    normal->z = int_as_float( p2 );
    *t        = int_as_float( p3 ); 
    *iidx     = p4 ; 
    position->x = int_as_float( p5 );
    position->y = int_as_float( p6 );
    position->z = int_as_float( p7 );
 
}

static __forceinline__ __device__ void setPayload( float3 normal, float t, unsigned instance_idx, float3 position )
{
    optixSetPayload_0( float_as_int( normal.x ) );
    optixSetPayload_1( float_as_int( normal.y ) );
    optixSetPayload_2( float_as_int( normal.z ) );
    optixSetPayload_3( float_as_int( t ) );

    optixSetPayload_4( instance_idx );
    optixSetPayload_5( float_as_int( position.x ) );
    optixSetPayload_6( float_as_int( position.y ) );
    optixSetPayload_7( float_as_int( position.z ) );
}

__forceinline__ __device__ uchar4 make_color( const float3& normal, unsigned iidx )
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

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const float2      d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
            ) - 1.0f;

    const float3& origin = params.eye;
    const float3& U = params.U ;
    const float3& V = params.V ;
    const float3& W = params.W ;
    const float& tmin = params.tmin ; 
    const float& tmax = params.tmax ; 
    const float3 direction   = normalize( d.x * U + d.y * V + W );

    float3   normal  = make_float3( 0.5f, 0.5f, 0.5f );
    float    t = 0.f ; 
    unsigned instance_idx = ~0u ; 
    float3   position = make_float3( 0.5f, 0.5f, 0.5f );

    trace( 
        params.handle,
        origin,
        direction,
        tmin,
        tmax,
        &normal, 
        &t, 
        &instance_idx,
        &position
    );

    uchar4 color = make_color( normal, instance_idx );
    unsigned index = idx.y * params.width + idx.x ;
    params.pixels[index] = color ; 
    params.isect[index] = make_float4( position.x, position.y, position.z, t ) ; 
}

extern "C" __global__ void __miss__ms()
{
    MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    float3    p = make_float3( 0.f, 0.f, 0.f); 
    unsigned instance_idx = 0 ; 
    float t_cand = 0.f ; 
    float3 normal = make_float3( rt_data->r, rt_data->g, rt_data->b );   
    float3 position = make_float3( 0.f, 0.f, 0.f ); 
    setPayload( normal,  t_cand, instance_idx, position );
}

extern "C" __global__ void __intersection__is()
{
    HitGroupData* hg_data  = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );
    const float3 orig = optixGetObjectRayOrigin();
    const float3 dir  = optixGetObjectRayDirection();
    const float  t_min = optixGetRayTmin() ; 

    const float3 center = {0.f, 0.f, 0.f};
    const float  radius = hg_data->values[0] ;
    const float3 O      = orig - center;
    const float3 D      = dir ; 
 
    float b = dot(O, D);
    float c = dot(O, O)-radius*radius;
    float d = dot(D, D);
    float disc = b*b-d*c;

    float sdisc = disc > 0.f ? sqrtf(disc) : 0.f ;   // ray has segment within sphere for sdisc > 0.f 
    float root1 = (-b - sdisc)/d ;
    float root2 = (-b + sdisc)/d ;  // root2 > root1 always

    float t_cand = sdisc > 0.f ? ( root1 > t_min ? root1 : root2 ) : t_min ; 
    bool valid_isect = t_cand > t_min ;

    if(valid_isect)
    {
        const float3 position = orig + t_cand*dir ;   
        const float3 shading_normal = ( O + t_cand*D )/radius;
        unsigned p0, p1, p2, p3;
        unsigned p4, p5, p6, p7;

        p0 = float_as_int( shading_normal.x );
        p1 = float_as_int( shading_normal.y );
        p2 = float_as_int( shading_normal.z );
        p3 = float_as_int( t_cand ) ; 

        p4 = float_as_int( position.x );
        p5 = float_as_int( position.y );
        p6 = float_as_int( position.z );
        p7 = float_as_int( t_cand ) ; 

        optixReportIntersection(
                t_cand,      
                0,          // user hit kind
                p0, p1, p2, p3, 
                p4, p5, p6, p7
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

    float t = int_as_float( optixGetAttribute_3() ) ; 

    const float3 position =
        make_float3(
                int_as_float( optixGetAttribute_4() ),
                int_as_float( optixGetAttribute_5() ),
                int_as_float( optixGetAttribute_6() )
                );

    unsigned instance_idx = optixGetInstanceIndex() ;

    float3 normal = normalize( optixTransformNormalFromObjectToWorldSpace( shading_normal ) ) * 0.5f + 0.5f ;  

    setPayload( normal, t,  instance_idx, position );
}

