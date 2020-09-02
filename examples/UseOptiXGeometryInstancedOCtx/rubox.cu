/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

rtDeclareVariable(float3, boxmin, , );
rtDeclareVariable(float3, boxmax, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(uint4,  intersect_identity,   attribute intersect_identity, ); 
rtDeclareVariable(uint4,  identity,  ,);


static __device__ float3 boxnormal(float t, float3 t0, float3 t1)
{
  float3 neg = make_float3(t==t0.x?1:0, t==t0.y?1:0, t==t0.z?1:0);
  float3 pos = make_float3(t==t1.x?1:0, t==t1.y?1:0, t==t1.z?1:0);
  return pos-neg;
}

RT_PROGRAM void rubox_intersect(int primIdx)
{
  float3 mn = boxmin*(1.f - float(primIdx)*0.01f ) ; 
  float3 mx = boxmax*(1.f - float(primIdx)*0.01f ) ; 

  float3 t0 = (mn - ray.origin)/ray.direction;
  float3 t1 = (mx - ray.origin)/ray.direction;
  float3 near = fminf(t0, t1);
  float3 far = fmaxf(t0, t1);
  float tmin = fmaxf( near );
  float tmax = fminf( far );

  if(tmin <= tmax) {
    bool check_second = true;
    if( rtPotentialIntersection( tmin ) ) {
       texcoord = make_float3( 0.0f );
       shading_normal = geometric_normal = boxnormal( tmin, t0, t1 );
       intersect_identity = identity ;  
       if(rtReportIntersection(0))
         check_second = false;
    } 
    if(check_second) {
      if( rtPotentialIntersection( tmax ) ) {
        texcoord = make_float3( 0.0f );
        shading_normal = geometric_normal = boxnormal( tmax, t0, t1 );
        intersect_identity = identity ;  
        rtReportIntersection(0);
      }
    }
  }
}

RT_PROGRAM void rubox_bounds (int primIdx, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  float3 mn = boxmin*(1.f - float(primIdx)*0.01f ) ; 
  float3 mx = boxmax*(1.f - float(primIdx)*0.01f ) ; 

  aabb->set(mn, mx);

  rtPrintf("//rubox_bounds primIdx %d mn(%10.3f %10.3f %10.3f) mx(%10.3f %10.3f %10.3f) \n", 
         primIdx,
         mn.x,mn.y,mn.z,  
         mx.x,mx.y,mx.z );  

}
