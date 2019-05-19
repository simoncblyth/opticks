#include <optix_world.h>
#include "helpers.h"  // make_color
#include "color_lookup.h"
#include "PerRayData_radiance.h"

#define OPTIX_VERSION_MAJOR (OPTIX_VERSION / 10000)
#define OPTIX_VERSION_MINOR ((OPTIX_VERSION % 10000) / 100)
#define OPTIX_VERSION_MICRO (OPTIX_VERSION % 100)


using namespace optix;

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        front, , );

rtDeclareVariable(float4,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(unsigned,      cameratype, , );

rtDeclareVariable(float,         pixeltimescale_cfg, , ) = 1e-10f;     // command line argument --pixeltimescale 
rtDeclareVariable(float,         pixeltime_scale, , );                 // adjustment that can be made from live GUI
rtDeclareVariable(unsigned,      pixeltime_style, , );

rtBuffer<uchar4, 2>              output_buffer;
//rtBuffer<float, 2>               depth_buffer;


rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  radiance_ray_type, , );
rtDeclareVariable(unsigned int,  resolution_scale, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );



rtDeclareVariable(unsigned int,  touch_mode, , );
rtDeclareVariable(unsigned int,  touch_bad, , );
rtDeclareVariable(uint2,         touch_index,  , );
rtDeclareVariable(uint2,         touch_dim,  , );
rtBuffer<uint4,2>         touch_buffer;


#define RED    make_uchar4(255u,  0u,  0u,255u)
#define GREEN  make_uchar4(  0u,255u,  0u,255u)
#define BLUE   make_uchar4(  0u,  0u,255u,255u)




RT_PROGRAM void pinhole_camera()
{

  PerRayData_radiance prd;
  prd.flag = 0u ; 
  prd.result = bad_color ;

  float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f ;  // (-1:1, -1:1 )


  optix::Ray ray ;

  if( cameratype == 0u ) // PERSPECTIVE_CAMERA
  {
      float3 ray_origin    = eye                          ; 
      float3 ray_direction = normalize(d.x*U + d.y*V + W) ;
      ray = optix::make_Ray( ray_origin , ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX) ;
  } 
  else if ( cameratype == 1u )  // ORTHOGRAPHIC_CAMERA
  {
      float3 ray_origin    = eye + d.x*U + d.y*V ; 
      float3 ray_direction = normalize(W)        ;
      ray = optix::make_Ray( ray_origin , ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX) ;
  }  
  else if ( cameratype == 2u ) // EQUIRECTANGULAR_CAMERA
  {
      // OptiX/SDK/optixTutorial/tutorial11.cu:env_camera
      // https://www.shadertoy.com/view/XsBSDR
      //
      //
      // azimuthal angle "phi"   :  -pi   -> pi
      // polar angle     "theta" :  -pi/2 -> pi/2

      float2 azipol = make_float2(launch_index) / make_float2(launch_dim) * make_float2(2.0f * M_PIf , M_PIf) + make_float2(M_PIf, 0.0f );
 
      //float2 azipol = make_float2(launch_index) / make_float2(launch_dim) * make_float2(2.0f*M_PIf , M_PIf ) ;
      //float2 azipol = d * make_float2(M_PIf , M_PIf/2.0f ) ;   // <- puts most distortion along horizontal center line
      float3 angle = make_float3(cos(azipol.x) * sin(azipol.y), -cos(azipol.y), sin(azipol.x) * sin(azipol.y));

      //                     cos(azi) sin(pol) , -cos(pol),   sin(azi)cos(pol) 

      //float3 angle = make_float3( sin(azipol.y) * cos(azipol.x), sin(azipol.y) * sin(azipol.x),  cos(azipol.y) ) ;
      // conventional spherical to cartesian 

 
      float3 ray_origin    = eye ; 
      float3 ray_direction = normalize(angle.x*normalize(U) + angle.y*normalize(V) + angle.z*normalize(W));

      ray = optix::make_Ray( ray_origin , ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX) ;
  }




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


  clock_t t0 = clock(); 

#if OPTIX_VERSION_MAJOR >= 6
  RTvisibilitymask mask = RT_VISIBILITY_ALL ;
  //RTrayflags      flags = RT_RAY_FLAG_NONE ;  
  RTrayflags      flags = RT_RAY_FLAG_DISABLE_ANYHIT ;  
  rtTrace(top_object, ray, prd, mask, flags);
#else
  rtTrace(top_object, ray, prd);
#endif

  clock_t t1 = clock(); 

  float dt = t1 - t0 ;  
  float pixeltime     = dt * pixeltime_scale * pixeltimescale_cfg ;  // CLI: --pixeltimescale 1e-10   GUI: G/composition/pixeltime to adjust 

  float4 result = prd.result ;   
  if( pixeltime_style == 1u )
  {
      result.x = pixeltime ;  
      result.y = pixeltime ;  
      result.z = pixeltime ;  
      // must not touch the depth in w for visibility see cu/material1_radiance.cu
  } 
  uchar4 color = make_color( result ) ;  

#if OPTIX_VERSION_MAJOR >= 6
   //color.x = 0xff ;  
#endif

  rtPrintf("//pinhole_camera dt %10.3f pixeltime_scale %10.3f pixeltimescale_cfg %10.3g pixeltime %10.3f color (%3d %3d %3d %3d)  \n",
          dt, pixeltime_scale, pixeltimescale_cfg, pixeltime, color.x, color.y, color.z, color.w  ); 

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
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  output_buffer[launch_index] = make_color( bad_color );
}


