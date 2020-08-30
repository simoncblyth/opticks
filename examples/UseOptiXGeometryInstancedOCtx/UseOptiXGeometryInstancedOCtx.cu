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

#include <optix_world.h>
using namespace optix;

// from optixrap/cu/helpers.h

// Convert a float3 in [0,1)^3 to a uchar4 in [0,255]^4 -- 4th channel is set to 255
static __device__ __inline__ optix::uchar4 make_color(const optix::float3& c)
{
    return optix::make_uchar4( static_cast<unsigned char>(__saturatef(c.x)*255.99f),  // R 
                               static_cast<unsigned char>(__saturatef(c.y)*255.99f),  // G 
                               static_cast<unsigned char>(__saturatef(c.z)*255.99f),  // B 
                               255u);                                                 // A 
}

/*
static __device__ __inline__ optix::uchar4 make_color(const optix::float4& c)
{
    return optix::make_uchar4( static_cast<unsigned char>(__saturatef(c.x)*255.99f),   // R 
                               static_cast<unsigned char>(__saturatef(c.y)*255.99f),   // G
                               static_cast<unsigned char>(__saturatef(c.z)*255.99f),   // B 
                               static_cast<unsigned char>(__saturatef(c.w)*255.99f));  // A
}
*/


struct PerRayData_radiance
{
  float3 result;
  float  importance;
  int depth;
};


rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(unsigned,     radiance_ray_type, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtDeclareVariable(rtObject,      top_object, , );

rtBuffer<uchar4, 2>   output_buffer;



rtDeclareVariable(float3, shading_normal,   attribute shading_normal, );  
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );


rtDeclareVariable(optix::Ray,           raycur, rtCurrentRay, );
rtDeclareVariable(float,                  t, rtIntersectionDistance, );

rtDeclareVariable(int4,   tex_param_0, , );
rtDeclareVariable(int4,   tex_param_1, , );

RT_PROGRAM void raygen_texture_test()
{
    float2 d = make_float2(launch_index) / make_float2(launch_dim) ;  // 0->1

    int texture_id = tex_param_0.w ; 
    //int texture_id = tex_param_1.w ; 
    int layer = 0 ; 
    output_buffer[launch_index] = rtTex2DLayered<uchar4>( texture_id, d.x, d.y, layer );
}

RT_PROGRAM void raygen()
{
    PerRayData_radiance prd;
    prd.result = make_float3( 1.f, 0.f, 0.f ) ;

    float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f ;   // -1:1

    optix::Ray ray = optix::make_Ray( eye, normalize(d.x*U + d.y*V + W), radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX) ; 
    rtTrace(top_object, ray, prd);

     //rtPrintf("//raygen launch_index.x %u launch_index.y %u launch_dim.x %u launch_dim.y %u \n", launch_index.x , launch_index.y, launch_dim.x , launch_dim.y   );
    output_buffer[launch_index] = make_color( prd.result ) ; 

    // make_uchar4(  255u, 0u, 0u,255u) ;  // red  (was expecting BGRA get RGBA)
}

// Returns shading normal as the surface shading result
RT_PROGRAM void closest_hit_radiance0()
{
  float3 isect = raycur.origin + t*raycur.direction ; 
  const float3 local = rtTransformPoint( RT_WORLD_TO_OBJECT, isect );  
  prd_radiance.result = normalize(local)*0.5f + 0.5f ; 
  //prd_radiance.result = normalize(isect)*0.5f + 0.5f ;    // coloring clearly global like this

  //prd_radiance.result = normalize(rtTransformNormal(RT_WORLD_TO_OBJECT, shading_normal))*0.5f + 0.5f;
  //prd_radiance.result = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal))*0.5f + 0.5f;
}


/**

layer:0 
    works fine

layer:1
    bizarre appearance, with messup that changes on each run
    as if texture reads are reading from non-intended GPU memory 

    * greyscale looks right throughout 
    * colored bands are messed up but not across the entire tex 


**/

RT_PROGRAM void closest_hit_textured()
{
  float3 isect = raycur.origin + t*raycur.direction ; 
  const float3 local = rtTransformPoint( RT_WORLD_TO_OBJECT, isect );  
  const float3 norm = normalize(local) ;  

  float f_theta = acos( norm.z )/M_PIf;                 // polar 0->pi ->  0->1
  float f_phi_ = atan2( norm.y, norm.x )/(2.f*M_PIf) ;  // azimuthal 0->2pi ->  0->1
  float f_phi = f_phi_ > 0.f ? f_phi_ : f_phi_ + 1.f ;  //  

  int texture_id = tex_param_0.w ; 
  //int texture_id = tex_param_1.w ; 
  int layer = 0 ; 
  uchar4 val = rtTex2DLayered<uchar4>( texture_id, f_phi, f_theta, layer );
  float3 result = make_float3( float(val.x)/255.99f,  float(val.y)/255.99f,  float(val.z)/255.99f ) ;   

  prd_radiance.result = result ;  ; 
}






RT_PROGRAM void miss()
{
  prd_radiance.result = make_float3(1.f, 1.f, 1.f) ;
}
RT_PROGRAM void printTest0()
{
     unsigned long long index = launch_index.x ;
     rtPrintf("//printTest0 d:%d launch_index.x %u launch_index.y %u launch_dim.x %u launch_dim.y %u \n", index, launch_index.x , launch_index.y, launch_dim.x , launch_dim.y   );
}
RT_PROGRAM void printTest1()
{
     unsigned long long index = launch_index.x ;
     rtPrintf("//printTest1 llu:%llu launch_index.x %u launch_index.y %u launch_dim.x %u launch_dim.y %u \n", index, launch_index.x , launch_index.y, launch_dim.x , launch_dim.y   );
}
RT_PROGRAM void exception()
{
    //rtPrintExceptionDetails();
}




