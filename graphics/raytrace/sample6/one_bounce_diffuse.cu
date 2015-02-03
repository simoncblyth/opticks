
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
#include <optix_math.h>
#include "commonStructs.h"
#include "random.h"

rtTextureSampler<float4, 2>   diffuse_map;         
rtBuffer<unsigned int, 2>     rnd_seeds;

rtDeclareVariable(int, sqrt_diffuse_samples, ,);

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

struct PerRayData_radiance
{
  float3 result;
  float importance;
  int depth;
};

struct PerRayData_shadow
{
  float3 attenuation;
};

rtBuffer<BasicLight>                 lights;
rtDeclareVariable(int,               frame, , );
rtDeclareVariable(unsigned int,      radiance_ray_type, , );
rtDeclareVariable(unsigned int,      shadow_ray_type, , );
rtDeclareVariable(float,             scene_epsilon, , );
rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(rtObject,          top_shadower, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow, rtPayload, );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );

RT_PROGRAM void closest_hit_radiance()
{
  float3 hit_point = ray.origin + t_hit * ray.direction;

  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 ffnormal               = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
  float2 uv                     = make_float2(texcoord);

  float3 Kd = make_float3(tex2D(diffuse_map, uv.x, uv.y));
  float3 result = make_float3(0);

  // Compute indirect bounce
  if(prd.depth < 1) {
    optix::Onb onb(ffnormal);
    unsigned int seed = rot_seed( rnd_seeds[ launch_index ], frame );
    const float inv_sqrt_samples = 1.0f / float(sqrt_diffuse_samples);

    int nx = sqrt_diffuse_samples;
    int ny = sqrt_diffuse_samples;
    while(ny--) {
      while(nx--) {
        // Stratify samples via simple jitterring
        float u1 = (float(nx) + rnd( seed ) )*inv_sqrt_samples;
        float u2 = (float(ny) + rnd( seed ) )*inv_sqrt_samples;

        float3 dir;
        optix::cosine_sample_hemisphere(u1, u2, dir);
        onb.inverse_transform(dir);

        PerRayData_radiance radiance_prd;
        radiance_prd.importance = prd.importance * optix::luminance(Kd);
        radiance_prd.depth = prd.depth + 1;

        if(radiance_prd.importance > 0.001f) {
          optix::Ray radiance_ray = optix::make_Ray(hit_point, dir, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
          rtTrace(top_object, radiance_ray, radiance_prd);

          result += radiance_prd.result;
        }
      }
      nx = sqrt_diffuse_samples;
    } 
    result *= (Kd)/((float)(M_PIf*sqrt_diffuse_samples*sqrt_diffuse_samples));
  }

  // Compute direct lighting
  int num_lights = lights.size();
  while(num_lights--) {
    const BasicLight& light = lights[num_lights];
    float3 L = light.pos - hit_point;
    float Ldist = length(light.pos - hit_point);
    L /= Ldist;
    float nDl = dot( ffnormal, L);

    if(nDl > 0.f) {
      if(light.casts_shadow) {
        PerRayData_shadow shadow_prd;
        shadow_prd.attenuation = make_float3(1.f);
        optix::Ray shadow_ray = optix::make_Ray( hit_point, L, shadow_ray_type, scene_epsilon, Ldist );
        rtTrace(top_shadower, shadow_ray, shadow_prd);

        if(fmaxf(shadow_prd.attenuation) > 0.f) {
          result += Kd * nDl * light.color * shadow_prd.attenuation;
        }
      }
    }
  }

  prd.result = result;
}

RT_PROGRAM void any_hit_shadow()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = make_float3(0);
  rtTerminateRay();
}
