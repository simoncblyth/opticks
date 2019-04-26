#include <optix_world.h>
#include "helpers.h"  // make_color
#include "color_lookup.h"
#include "PerRayData_radiance.h"

using namespace optix;

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        front, , );

rtDeclareVariable(float4,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(unsigned int,  parallel, , );

rtBuffer<uchar4, 2>              output_buffer;
//rtBuffer<float, 2>               depth_buffer;


rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  radiance_ray_type, , );
rtDeclareVariable(unsigned int,  resolution_scale, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtDeclareVariable(float, timetracerscale, , ) = 1e-6f;


rtDeclareVariable(unsigned int,  touch_mode, , );
rtDeclareVariable(unsigned int,  touch_bad, , );
rtDeclareVariable(uint2,         touch_index,  , );
rtDeclareVariable(uint2,         touch_dim,  , );
rtBuffer<uint4,2>         touch_buffer;


#define RED    make_uchar4(255u,  0u,  0u,255u)
#define GREEN  make_uchar4(  0u,255u,  0u,255u)
#define BLUE   make_uchar4(  0u,  0u,255u,255u)



RT_PROGRAM void pinhole_camera_timetracer()
{

  PerRayData_radiance prd;
  prd.flag = 0u ; 
  prd.result = bad_color ;

  float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f ;

  optix::Ray ray = parallel == 0 ? 
                       optix::make_Ray( eye                 , normalize(d.x*U + d.y*V + W), radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX)
                     :
                       optix::make_Ray( eye + d.x*U + d.y*V , normalize(W)                , radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX)
                     ;

  clock_t t0 = clock(); 

  rtTrace(top_object, ray, prd);

  clock_t t1 = clock(); 

  float dt = ( t1 - t0 ) ; 

  float pixel_time  = dt * timetracerscale ;

  uchar4  color = make_color( make_float3( pixel_time )); 

  rtPrintf("//pinhole_camera_timetracer dt %10.3f pixel_time %10.3f timetracerscale %10.3g color (%d %d %d) \n", dt, pixel_time, timetracerscale, color.x, color.y, color.z );  

  //uchar4  color = RED ;  

  output_buffer[launch_index] = color ; 
}


RT_PROGRAM void pinhole_camera()
{
  PerRayData_radiance prd;
  prd.flag = 0u ; 
  prd.result = bad_color ;

  float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f ;

  optix::Ray ray = parallel == 0 ? 
                       optix::make_Ray( eye                 , normalize(d.x*U + d.y*V + W), radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX)
                     :
                       optix::make_Ray( eye + d.x*U + d.y*V , normalize(W)                , radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX)
                     ;


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
  // The result of this is that ray traced "near" clipping tends to 
  // make circular-ish cuts into geometry.
  //


#if RAYTRACE_TIMEVIEW
  clock_t t0 = clock(); 
#endif

  rtTrace(top_object, ray, prd);

#if RAYTRACE_TIMEVIEW
  clock_t t1 = clock(); 
  float pixel_time     = ( t1 - t0 ) * time_view_scale ;
  uchar4  color = make_color( make_float3( pixel_time )); 
#else
  uchar4 color = make_color( prd.result ) ; // BGRA
#endif

  if( resolution_scale == 1)  
  { 
      output_buffer[launch_index] = color ; 
  }
  else if( resolution_scale == 2)
  {
      unsigned int wx2 = 2*launch_index.x ; 
      unsigned int wy2 = 2*launch_index.y ; 

      uint2 idx00 = make_uint2(wx2  , wy2) ; 
      uint2 idx10 = make_uint2(wx2+1, wy2) ; 
      uint2 idx01 = make_uint2(wx2  , wy2+1) ; 
      uint2 idx11 = make_uint2(wx2+1, wy2+1) ; 

      output_buffer[idx00] = color ; 
      output_buffer[idx10] = color ; 
      output_buffer[idx01] = color ; 
      output_buffer[idx11] = color ; 

  }
  else if( resolution_scale > 2)
  {
      unsigned int wx = resolution_scale*launch_index.x ; 
      unsigned int wy = resolution_scale*launch_index.y ; 
      for(unsigned int i=0 ; i < resolution_scale ; i++){
      for(unsigned int j=0 ; j < resolution_scale ; j++){
          uint2 idx = make_uint2(wx+i, wy+j) ; 
          output_buffer[idx] = color ; 
      }
      }
  }
}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  //rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  output_buffer[launch_index] = make_color( bad_color );
}


