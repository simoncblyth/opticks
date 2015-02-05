
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

#include <optix_world.h>
#include "commonStructs.h"
#include "helpers.h"

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


rtDeclareVariable(int,               max_depth, , );
rtBuffer<BasicLight>                 lights;
rtDeclareVariable(float3,            ambient_light_color, , );
rtDeclareVariable(unsigned int,      radiance_ray_type, , );
rtDeclareVariable(unsigned int,      shadow_ray_type, , );
rtDeclareVariable(float,             scene_epsilon, , );
rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(rtObject,          top_shadower, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow, rtPayload, );

static __device__ void phongShadowed()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = optix::make_float3(0);
  rtTerminateRay();
}

static
__device__ void phongShade( float3 p_Kd,
                            float3 p_Ka,
                            float3 p_Ks,
                            float3 p_normal,
                            float  p_phong_exp,
                            float3 p_reflectivity )
{
  float3 hit_point = ray.origin + t_hit * ray.direction;
  
  float3 result = p_Ka * ambient_light_color; // ambient contribution

  unsigned int num_lights = lights.size();
  for(int i = 0; i < num_lights; ++i) // compute direct lighting
  {
      BasicLight light = lights[i];
      float Ldist = optix::length(light.pos - hit_point);
      float3 L = optix::normalize(light.pos - hit_point);
      float nDl = optix::dot( p_normal, L);

      float3 light_attenuation = make_float3(static_cast<float>( nDl > 0.0f ));

      if ( nDl > 0.0f && light.casts_shadow ) // cast shadow ray
      {
          PerRayData_shadow shadow_prd;
          shadow_prd.attenuation = make_float3(1.0f);
          optix::Ray shadow_ray = optix::make_Ray( hit_point, L, shadow_ray_type, scene_epsilon, Ldist );
          rtTrace(top_shadower, shadow_ray, shadow_prd);
          light_attenuation = shadow_prd.attenuation;
      }

      if( fmaxf(light_attenuation) > 0.0f ) // If not completely shadowed, light the hit point
      {
          float3 Lc = light.color * light_attenuation;

          result += p_Kd * nDl * Lc;

          float3 H = optix::normalize(L - ray.direction);
          float nDh = optix::dot( p_normal, H );
          if(nDh > 0) 
          {
              float power = pow(nDh, p_phong_exp);
              result += p_Ks * power * Lc;
          }
      }
  }

  if( fmaxf( p_reflectivity ) > 0 ) 
  {
      PerRayData_radiance new_prd; // ray tree attenuation
      new_prd.importance = prd.importance * optix::luminance( p_reflectivity );
      new_prd.depth = prd.depth + 1;

      if( new_prd.importance >= 0.01f && new_prd.depth <= max_depth) // reflection ray
      {
          float3 R = optix::reflect( ray.direction, p_normal );
          optix::Ray refl_ray = optix::make_Ray( hit_point, R, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX );
          rtTrace(top_object, refl_ray, new_prd);
          result += p_reflectivity * new_prd.result;
      }
  }
  
  // pass the color back up the tree
  prd.result = result;
  //prd.result = make_float3(0.f, 1.f, 0.f); // green geometry silhouette against blue bkgd, on entering geometry plain green
}
