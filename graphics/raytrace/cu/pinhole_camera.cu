#include <optix_world.h>
#include "helpers.h"

#include "RayTraceConfigInc.h"
#if RAYTRACE_CURAND
#include <curand_kernel.h>
#endif

#include "PerRayData_radiance.h"


using namespace optix;


rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtBuffer<uchar4, 2>              output_buffer;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  radiance_ray_type, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;



#define RAYTRACE_TOUCH
#ifdef RAYTRACE_TOUCH
rtDeclareVariable(unsigned int,  touch_mode, , );
rtDeclareVariable(unsigned int,  bad_touch, , );
rtDeclareVariable(unsigned int,  touch_ray_type, , );
rtDeclareVariable(uint2, touch_index,  , );
rtDeclareVariable(uint2, touch_dim,  , );

rtBuffer<unsigned int,2>           touch_buffer;
#endif


#if RAYTRACE_CURAND
rtBuffer<curandState, 1> rng_states ;
#endif





// whilst viewing output_buffer from pinhole_camera touching 
// a pixel yields the below touch params which are used
// here to shoot a touch ray to find the object under the pixel     
//
// touch_index : touched pixel coordinates 
// touch_dim   : pixel dimensions with 
//
// touch pixel coordinates into  [ -1 : 1, -1 : 1 ]
//

/* 
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, touch_ray_type, scene_epsilon, RT_DEFAULT_MAX);

  PerRayData_touch prd;
  prd.depth = 0;
  prd.node = bad_touch ;  

  rtTrace(top_object, ray, prd);

  touch_buffer[launch_index] = prd.node ;  // returning the index of the node touched

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

*/



RT_PROGRAM void pinhole_camera()
{
#if RAYTRACE_TIMEVIEW
  clock_t t0 = clock(); 
#endif

  //
  // touch_mode launches a single ray corresponding to the pixel under the mouse
  // with (launch_index 0,0 launch_dim 1,1) allowing to feel which 
  // geometry node resulted in a pixel from the prior standard launch
  //
  // pixel coordinates -> normalized coordinates  [ -1:1, -1:1 ]
  //

  float2 d = touch_mode ?  
                     make_float2(touch_index) / make_float2(touch_dim) * 2.f - 1.f 
                   : 
                     make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;
                   ;


  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;
  prd.node = bad_touch ;

#if RAYTRACE_CURAND
  unsigned long long id = launch_index.x + launch_dim.x * launch_index.y ; 
  prd.rng = rng_states[id];
#endif


  rtTrace(top_object, ray, prd);

#if RAYTRACE_CURAND
  //prd.result.x = curand_uniform(&prd.rng); 
  rng_states[id] = prd.rng ; 
#endif


#if RAYTRACE_TIMEVIEW
  clock_t t1 = clock(); 
 
  float expected_fps   = 1.0f;
  float pixel_time     = ( t1 - t0 ) * time_view_scale * expected_fps;
  output_buffer[launch_index] = make_color( make_float3(  pixel_time ) ); 
#else
  //if(launch_index.x == 0 && launch_index.y == 0) touch_buffer[launch_index] = prd.node ;  // returning the index of the node touched
  output_buffer[launch_index] = make_color( prd.result );
  //output_buffer[launch_index] = make_color(make_float3(0.5f));   // plain grey screen, not silhouette : all pixels go this way 

  if(touch_mode)
  {
      touch_buffer[launch_index] = prd.node ;  // returning the index of the node touched
      rtPrintf("pinhole_camera.cu::pinhole_camera  node %d \n", prd.node );
  }

#endif
}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  //rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  output_buffer[launch_index] = make_color( bad_color );
}





