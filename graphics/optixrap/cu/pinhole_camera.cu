#include <optix_world.h>
#include "helpers.h"  // make_color
#include "color_lookup.h"
#include "hemi-pmt.h"
#include "PerRayData_radiance.h"

using namespace optix;


rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float4,        ZProj, , );

rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(unsigned int,  parallel, , );

rtBuffer<uchar4, 2>              output_buffer;
//rtBuffer<float, 2>               depth_buffer;


rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  radiance_ray_type, , );
rtDeclareVariable(unsigned int,  resolution_scale, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(float, t,            rtIntersectionDistance, );

rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;


rtDeclareVariable(unsigned int,  touch_mode, , );
rtDeclareVariable(unsigned int,  touch_bad, , );
rtDeclareVariable(uint2,         touch_index,  , );
rtDeclareVariable(uint2,         touch_dim,  , );
rtBuffer<uint4,2>         touch_buffer;

// BGRA
#define BLUE  make_uchar4(255u,  0u,  0u,255u)
#define GREEN make_uchar4(  0u,255u,  0u,255u)
#define RED   make_uchar4(  0u,  0u,255u,255u)


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

  //color_dump(); 


  PerRayData_radiance prd;
  prd.flag = 0u ; 
  prd.result = bad_color ;

  float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f ;

 
  float3 ray_origin ;
  float3 ray_direction ;   
  float3 front = normalize(W) ;

  if(parallel == 0)
  {
      ray_origin = eye;
      ray_direction = normalize(d.x*U + d.y*V + W);   
  }
  else
  {
      ray_origin    = eye + d.x*U + d.y*V ;
      ray_direction = normalize(W) ; 
  }  


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
  // unsigned long long id = launch_index.x + launch_dim.x * launch_index.y ; 


  rtTrace(top_object, ray, prd);

  float zHit_eye = -t*dot(front, ray_direction) ;   // intersect z coordinate (eye frame), always -ve 
  float zHit_ndc = parallel == 0 ? -ZProj.z - ZProj.w/zHit_eye : ZProj.z*zHit_eye + ZProj.w ;  // should be in range -1:1 for visibles
  float zHit_clip = 0.5f*zHit_ndc + 0.5f ;   // 0:1 for visibles

  //rtPrintf("pinhole_camera t %10.4f zHit_eye %10.4f  ZProj.z %10.4f ZProj.w %10.4f zHit_ndc %10.4f zHit_clip %10.4f \n", t, zHit_eye, ZProj.z, ZProj.w , zHit_ndc, zHit_clip );


#if RAYTRACE_TIMEVIEW
  clock_t t1 = clock(); 
  float expected_fps   = 1.0f;
  float pixel_time     = ( t1 - t0 ) * time_view_scale * expected_fps;
  uchar4  color = = make_color( make_float3(  pixel_time ) ); 
#else
  // BGRA
  uchar4 color = prd.flag == HP_QCAP_O ? RED :  make_color( prd.result );

  //if(prd.flag > 0u)
  //   rtPrintf("prd.flag %u color %u %u %u %u \n", prd.flag, color.x, color.y, color.z, color.w ); 


#endif


  if( resolution_scale == 1)  
  { 
      output_buffer[launch_index] = color ; 
      //depth_buffer[launch_index] = zHit_clip ; 
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

      //depth_buffer[idx00] = zHit_clip ; 
      //depth_buffer[idx10] = zHit_clip ; 
      //depth_buffer[idx01] = zHit_clip ; 
      //depth_buffer[idx11] = zHit_clip ; 

  }
  else if( resolution_scale > 2)
  {
      unsigned int wx = resolution_scale*launch_index.x ; 
      unsigned int wy = resolution_scale*launch_index.y ; 
      for(unsigned int i=0 ; i < resolution_scale ; i++){
      for(unsigned int j=0 ; j < resolution_scale ; j++){
          uint2 idx = make_uint2(wx+i, wy+j) ; 
          output_buffer[idx] = color ; 
          //depth_buffer[idx] = zHit_clip ; 
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





