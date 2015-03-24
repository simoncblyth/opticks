#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "helpers.h"

using namespace optix;

#include "PerRayData_radiance.h"


rtDeclareVariable(float3,        eye, , );        // center of film plane
rtDeclareVariable(float3,        U, , );          // horizontal orientation; len(U) specifies width of film plane
rtDeclareVariable(float3,        V, , );          // vertical orientation;   len(V) specifies height of film plane
rtDeclareVariable(float3,        W, , );          // view direction 
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtBuffer<uchar4, 2>              output_buffer;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  radiance_ray_type, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );


#if 0
rtDeclareVariable(unsigned int,  touch_mode, , );
rtDeclareVariable(unsigned int,  touch_bad, , );
rtDeclareVariable(uint2,         touch_index,  , );
rtDeclareVariable(uint2,         touch_dim,  , );
rtBuffer<unsigned int,2>         touch_buffer;
#endif


RT_PROGRAM void orthographic_camera()
{
  size_t2 screen = output_buffer.size();

  //float2 d = touch_mode ?  
  //                   make_float2(touch_index) / make_float2(touch_dim) * 2.f - 1.f 
  //                 : 
  //                   make_float2(launch_index) / make_float2(screen) * 2.f - 1.f  // film coords
  //                 ;


  float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f  ; // film coords
  float3 ray_origin    = eye + d.x*U + d.y*V;                              // eye + offset in film space
  float3 ray_direction = W;                                                // always parallel view direction
  
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX );

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;

  rtTrace(top_object, ray, prd);

  output_buffer[launch_index] = make_color( prd.result );

  //if(touch_mode)
  //{
  //    touch_buffer[launch_index] = prd.node ;  // returning the index of the node touched
  //    rtPrintf("orthographic_camera.cu::orthographic_camera  node %d \n", prd.node );
  //}

}




RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  output_buffer[launch_index] = make_color( bad_color );
}
