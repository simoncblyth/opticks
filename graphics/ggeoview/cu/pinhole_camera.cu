#include <optix_world.h>
#include "helpers.h"  // make_color

//#include "RayTraceConfigInc.h"
#include <curand_kernel.h>

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


rtDeclareVariable(unsigned int,  touch_mode, , );
rtDeclareVariable(unsigned int,  touch_bad, , );
rtDeclareVariable(uint2,         touch_index,  , );
rtDeclareVariable(uint2,         touch_dim,  , );
rtBuffer<unsigned int,2>         touch_buffer;


//rtBuffer<curandState, 1> rng_states ;



RT_PROGRAM void pinhole_camera()
{
#if RAYTRACE_TIMEVIEW
  clock_t t0 = clock(); 
#endif

  //
  // touch_mode launches a single ray corresponding to the pixel under the mouse
  // with (launch_index 0,0 launch_dim 1,1) allowing to feel which 
  // geometry node resulted in a pixel from the prior standard non-touch launch
  //
  // pixel coordinates -> normalized coordinates  [ -1:1, -1:1 ]
  //

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;
  prd.node = touch_bad ;
  prd.result = bad_color ;

  float2 d = touch_mode ?  
                     make_float2(touch_index) / make_float2(touch_dim) * 2.f - 1.f 
                   : 
                     make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f
                   ;

   
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);   
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

  // (d.x,d.y) spans screen pixels (-1:1,-1:1) 
  //
  // oglrap-/Composition::getEyeUVW provides : eye,U,V,W
  //
  //   * eye : world frame position of camera  
  //   * U,V : vectors defining dimension and direction of x,y 
  //           axes of image plane expresses in world frame
  //   * W   : is eye to image plane direction and dimension   
  // 
  // scene_epsilon is "t_min" but ray_direction is normalized, 
  // so that makes "t_min" a world frame quantity, which makes 
  // setting it to world frame camera Near to be appropriate 
  // (when looking straight ahead at least). 
  //
  // They are not really equivalent though, near being the distance
  // to the screen parallel frustum plane whereas 
  // scene_epsilon is the distance along the ray at which to start 
  // accepting intersections ?
  //
  //
  unsigned long long id = launch_index.x + launch_dim.x * launch_index.y ; 
  //prd.rng = rng_states[id];

  rtTrace(top_object, ray, prd);

  ////prd.result.x = curand_uniform(&prd.rng); 
  //rng_states[id] = prd.rng ; 


#if RAYTRACE_TIMEVIEW
  clock_t t1 = clock(); 
 
  float expected_fps   = 1.0f;
  float pixel_time     = ( t1 - t0 ) * time_view_scale * expected_fps;
  output_buffer[launch_index] = make_color( make_float3(  pixel_time ) ); 
#else
  output_buffer[launch_index] = make_color( prd.result );

  if(touch_mode)
  {
      touch_buffer[launch_index] = prd.node ;  // returning the index of the node touched
      rtPrintf("pinhole_camera.cu::pinhole_camera  node %d \n", prd.node );

      // cannot do wavelength lookups here, as wavelength_texture 
      // not defined in scopes: Program, Context
      // instead must do in closest_hit where Material scope
      // is available 
  }

#endif
}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  //rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  output_buffer[launch_index] = make_color( bad_color );
}





