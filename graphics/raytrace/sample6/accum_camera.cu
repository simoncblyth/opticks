
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "helpers.h"
#include "random.h"

using namespace optix;

struct PerRayData_radiance
{
  float3 result;
  float  importance;
  int    depth;
};

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtBuffer<uchar4, 2>              output_buffer;
rtBuffer<float4, 2>              accum_buffer;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  radiance_ray_type, , );
rtDeclareVariable(int,           frame, , );
rtDeclareVariable(float,         jitter_factor, ,) = 0.0f;

rtDeclareVariable(uint2,         launch_index, rtLaunchIndex, );
rtDeclareVariable(float,         time_view_scale, , ) = 1e-6f;

rtBuffer<unsigned int, 2>        rnd_seeds;

//#define TIME_VIEW


RT_PROGRAM void pinhole_camera()
{
#ifdef TIME_VIEW
  clock_t t0 = clock(); 
#endif
  size_t2 screen = output_buffer.size();

  // Subpixel jitter: send the ray through a different position inside the pixel each time,
  // to provide antialiasing.
  unsigned int seed = rot_seed( rnd_seeds[ launch_index ], frame );
  float2 subpixel_jitter = make_float2(rnd( seed ) - 0.5f, rnd( seed ) - 0.5f) * jitter_factor;

  float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  
  optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon );

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;

  rtTrace(top_object, ray, prd);

#ifdef TIME_VIEW
  clock_t t1 = clock();
 
  float expected_fps   = 1.0f;
  float pixel_time     = ( t1 - t0 ) * time_view_scale * expected_fps;
  output_buffer[launch_index] = make_color( make_float3(  pixel_time ) );
#else
  float4 acc_val = accum_buffer[launch_index];
  if( frame > 0 )
    acc_val = lerp( acc_val, make_float4( prd.result, 0.f), 1.0f / static_cast<float>( frame+1 ) );
  else
    acc_val = make_float4(prd.result, 0.f);
  output_buffer[launch_index] = make_color( make_float3( acc_val ) );
  accum_buffer[launch_index] = acc_val;
#endif
}


RT_PROGRAM void orthographic_camera()
{
  size_t2 screen = output_buffer.size();

  // Subpixel jitter: send the ray through a different position inside the pixel each time,
  // to provide antialiasing.
  unsigned int seed = rot_seed( rnd_seeds[ launch_index ], frame );
  float2 subpixel_jitter = make_float2(rnd( seed ) - 0.5f, rnd( seed ) - 0.5f) * jitter_factor;

  float2 d = (make_float2(launch_index) + subpixel_jitter)
      / make_float2(screen) * 2.f - 1.f;                                   // film coords
  float3 ray_origin    = eye + d.x*U + d.y*V;                              // eye + offset in film space
  float3 ray_direction = W;                                                // always parallel view direction

  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;

  rtTrace(top_object, ray, prd);

  float4 acc_val = accum_buffer[launch_index];
  if( frame > 0 )
    acc_val += make_float4(prd.result, 0.f);
  else
    acc_val = make_float4(prd.result, 0.f);
  output_buffer[launch_index] = make_color( make_float3(acc_val) * (1.0f/float(frame+1)) );
  accum_buffer[launch_index] = acc_val;
}



RT_PROGRAM void exception()
{
  output_buffer[launch_index] = make_color( bad_color );
}
