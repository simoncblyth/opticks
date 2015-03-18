#include <optix_world.h>
#include "helpers.h"

using namespace optix;

#include "PerRayData.h"


rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(unsigned int,  bad_touch, , );

rtDeclareVariable(rtObject,      top_object, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtDeclareVariable(unsigned int,  touch_ray_type, , );

//rtBuffer<uchar4, 2>              touch_buffer;
rtBuffer<unsigned int,2>           touch_buffer;


rtDeclareVariable(uint2, touch_index,  , );
rtDeclareVariable(uint2, touch_dim,  , );



RT_PROGRAM void touch_pinhole_camera()
{
  // whilst viewing output_buffer from pinhole_camera touching 
  // a pixel yields the below touch params which are used
  // here to shoot a touch ray to find the object under the pixel     
  //
  // touch_index : touched pixel coordinates 
  // touch_dim   : pixel dimensions with 
  //
  // touch pixel coordinates into  [ -1 : 1, -1 : 1 ]
  //
  float2 d = make_float2(touch_index) / make_float2(touch_dim) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, touch_ray_type, scene_epsilon, RT_DEFAULT_MAX);

  PerRayData_touch prd;
  prd.depth = 0;
  prd.node = bad_touch ;  

  rtTrace(top_object, ray, prd);

  touch_buffer[launch_index] = prd.node ;

  // only touch single pixels, so can be verbose here
  rtPrintf("touch_pinhole_camera.cu::touch_pinhole_camera  node %d \n", prd.node );

  rtPrintf("touch_pinhole_camera.cu::touch_pinhole_camera  texlookup_b  %10.3f %10.3f %10.3f %10.3f \n", 
     prd.texlookup_b.x,
     prd.texlookup_b.y,
     prd.texlookup_b.z,
     prd.texlookup_b.w );

  rtPrintf("touch_pinhole_camera.cu::touch_pinhole_camera  texlookup_g  %10.3f %10.3f %10.3f %10.3f \n", 
     prd.texlookup_g.x,
     prd.texlookup_g.y,
     prd.texlookup_g.z,
     prd.texlookup_g.w );

  rtPrintf("touch_pinhole_camera.cu::touch_pinhole_camera  texlookup_r  %10.3f %10.3f %10.3f %10.3f \n", 
     prd.texlookup_r.x,
     prd.texlookup_r.y,
     prd.texlookup_r.z,
     prd.texlookup_r.w );

}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  touch_buffer[launch_index] = bad_touch ;
}





