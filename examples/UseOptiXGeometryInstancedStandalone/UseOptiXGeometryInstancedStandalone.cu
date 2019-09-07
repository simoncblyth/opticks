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
    return optix::make_uchar4( static_cast<unsigned char>(__saturatef(c.z)*255.99f),  // B 
                               static_cast<unsigned char>(__saturatef(c.y)*255.99f),  // G 
                               static_cast<unsigned char>(__saturatef(c.x)*255.99f),  // R 
                               255u);                                                 // A 
}

/*
static __device__ __inline__ optix::uchar4 make_color(const optix::float4& c)
{
    return optix::make_uchar4( static_cast<unsigned char>(__saturatef(c.z)*255.99f),  // B 
                               static_cast<unsigned char>(__saturatef(c.y)*255.99f),  // G
                               static_cast<unsigned char>(__saturatef(c.x)*255.99f),  // R 
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


RT_PROGRAM void raygen()
{
    PerRayData_radiance prd;
    prd.result = make_float3( 1.f, 0.f, 0.f ) ;

    float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f ;

    optix::Ray ray = optix::make_Ray( eye, normalize(d.x*U + d.y*V + W), radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX) ; 
    rtTrace(top_object, ray, prd);

     //rtPrintf("//raygen launch_index.x %u launch_index.y %u launch_dim.x %u launch_dim.y %u \n", launch_index.x , launch_index.y, launch_dim.x , launch_dim.y   );
    output_buffer[launch_index] = make_color( prd.result ) ; 

    // make_uchar4(  255u, 0u, 0u,255u) ;  // red  (was expecting BGRA get RGBA)
}

// Returns shading normal as the surface shading result
RT_PROGRAM void closest_hit_radiance0()
{
  prd_radiance.result = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal))*0.5f + 0.5f;
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


/*
Commenting out "rtPrintExceptionDetails()" gets rid of the .f64 in the ptx

    [blyth@localhost UseOptiXGeometryInstancedStandalone]$ grep .f64 /tmp/blyth/opticks/UseOptiXGeometryInstancedStandalone/ptx/UseOptiXGeometryInstancedStandalone_generated_UseOptiXGeometryInstancedStandalone.cu.ptx
    [blyth@localhost UseOptiXGeometryInstancedStandalone]$ 

    [blyth@localhost UseOptiXGeometryInstancedStandalone]$ grep f64 /tmp/blyth/opticks/UseOptiXGeometryInstancedStandalone/ptx/UseOptiXGeometryInstancedStandalone_generated_UseOptiXGeometryInstancedStandalone.cu.ptx
        .reg .f64   %fd<9>;
        cvt.ftz.f64.f32 %fd1, %f1;
        cvt.ftz.f64.f32 %fd2, %f6;
        cvt.ftz.f64.f32 %fd3, %f3;
        cvt.ftz.f64.f32 %fd4, %f2;
        st.local.v2.f64     [%rd52+32], {%fd4, %fd3};
        cvt.ftz.f64.f32 %fd5, %f5;
        cvt.ftz.f64.f32 %fd6, %f4;
        st.local.v2.f64     [%rd52+48], {%fd6, %fd5};
        st.local.f64    [%rd52+64], %fd2;
        cvt.ftz.f64.f32 %fd7, %f8;
        cvt.ftz.f64.f32 %fd8, %f7;
        st.local.v2.f64     [%rd52+80], {%fd8, %fd7};
    [blyth@localhost UseOptiXGeometryInstancedStandalone]$ 



[blyth@localhost UseOptiXGeometryInstancedStandalone]$ grep .visible /tmp/blyth/opticks/UseOptiXGeometryInstancedStandalone/ptx/UseOptiXGeometryInstancedStandalone_generated_UseOptiXGeometryInstancedStandalone.cu.ptx | c++filt
.visible .entry raygen()(
.visible .entry closest_hit_radiance0()(
.visible .entry miss()(
.visible .entry printTest0()(
.visible .entry printTest1()(
.visible .entry exception()(
[blyth@localhost UseOptiXGeometryInstancedStandalone]$ 





*/





