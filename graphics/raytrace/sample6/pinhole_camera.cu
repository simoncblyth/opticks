#include <optix_world.h>
#include "helpers.h"

#include "RayTraceConfigInc.h"
#if RAYTRACE_CURAND
#include <curand_kernel.h>
#endif


using namespace optix;

struct PerRayData_radiance
{
  float3 result;
  float  importance;
  int    depth;
};

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


#if RAYTRACE_CURAND
rtBuffer<curandState, 1> rng_states ;
#endif


RT_PROGRAM void pinhole_camera()
{
#if RAYTRACE_TIMEVIEW
  clock_t t0 = clock(); 
#endif
  // pixel coordinates into  [ -1 : 1, -1 : 1 ]
  float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  
#if RAYTRACE_CURAND
  unsigned long long id = launch_index.x + launch_dim.x * launch_index.y ; 
  curandState rng = rng_states[id];
#endif

  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;

  rtTrace(top_object, ray, prd);


#if RAYTRACE_CURAND
  float u = curand_uniform(&rng); 
  prd.result.x = u ; 
  rng_states[id] = rng ; 
#endif


#if RAYTRACE_TIMEVIEW
  clock_t t1 = clock(); 
 
  float expected_fps   = 1.0f;
  float pixel_time     = ( t1 - t0 ) * time_view_scale * expected_fps;
  output_buffer[launch_index] = make_color( make_float3(  pixel_time ) ); 
#else
  output_buffer[launch_index] = make_color( prd.result );
  //output_buffer[launch_index] = make_color(make_float3(0.5f));   // plain grey screen, not silhouette : all pixels go this way 
#endif
}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  output_buffer[launch_index] = make_color( bad_color );
}





