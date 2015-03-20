//-----------------------------------------------------------------------------
//  adapted from /Developer/OptiX_301/SDK/sample6/sample6.cpp
// 
//  sample6.cpp: Renders an Obj model.
//  
//-----------------------------------------------------------------------------

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <sutil.h>
#include <GLUTDisplay.h>

#include "G4DAELoader.hh"
#include "GSolid.hh"
#include "GSubstance.hh"
#include "GGeo.hh"

#include "commonStructs.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>

#include "cu/random.h"

#include "RayTraceConfig.hh"
#include "MeshScene.h"

#include <stdlib.h>
#include <stdio.h>

#include "LaunchSequence.hh"
#include "cuRANDWrapper.hh"

using namespace optix;

#include "curand.h"
#include "curand_kernel.h"



enum RayType
{ 
   radiance_ray_type,
   shadow_ray_type
};



//------------------------------------------------------------------------------
//
// MeshViewer class 
//
//------------------------------------------------------------------------------
class MeshViewer : public MeshScene
{
public:
  //
  // Helper types
  //
  
  static unsigned int TOUCH_BAD ; 

  enum ShadeMode
  {
    SM_PHONG=0,
    SM_AO,
    SM_NORMAL,
    SM_ONE_BOUNCE_DIFFUSE,
    SM_AO_PHONG
  };

  enum CameraMode
  {
    CM_PINHOLE=0,
    CM_ORTHO
  };

  //
  // MeshViewer specific  
  //
  MeshViewer();

  // Setters for controlling application behavior
  void setShadeMode( ShadeMode mode )              { m_shade_mode = mode;               }
  void setCameraMode( CameraMode mode )            { m_camera_mode = mode;              }
  void setAORadius( float ao_radius )              { m_ao_radius = ao_radius;           }
  void setAOSampleMultiplier( int ao_sample_mult ) { m_ao_sample_mult = ao_sample_mult; }
  void setLightScale( float light_scale )          { m_light_scale = light_scale;       }
  void setAA( bool onoff )                         { m_aa_enabled = onoff;              }
  void setAnimation( bool anim )                   { m_animation = anim;                }

  void createDevRngStates(
      unsigned int elements, 
      unsigned long long seed=0, 
      unsigned long long offset=0,
      unsigned int max_blocks=128,
      unsigned int threads_per_block=256
    );

  void initDevRngStates(
      unsigned int elements, 
      unsigned long long seed=0, 
      unsigned long long offset=0,
      unsigned int       max_blocks=128,
      unsigned int       threads_per_block=256
    );
  void resizeDevRngStates(unsigned int elements, bool force=false);

  //
  // From SampleScene
  //
  virtual void   initScene( InitialCameraData& camera_data );
  virtual void   doResize( unsigned int width, unsigned int height );
  virtual void   trace( const RayGenCameraData& camera_data );
  virtual void   cleanUp();
  virtual bool   keyPressed(unsigned char key, int x, int y);
  virtual Buffer getOutputBuffer();

  void touch(unsigned char key, int x, int y);
  void dumpNode(unsigned int nodeIndex);
  void dumpCamera(const char* msg, unsigned char key, int x, int y ); 

public:
  optix::Context getContext();
  void setFile(const char* path ){ m_path = strdup(path) ; }
  char* getFile(){ return m_path ; } 

public:
  GGeo* getGGeo();
private:
  void setGGeo(GGeo* ggeo);  

private:
  void initContext();
  void initLights();
  void initMaterial();
  void initGeometry();
  void initCamera( InitialCameraData& cam_data );
  void preprocess();
  void resetAccumulation();
  void genRndSeeds( unsigned int width, unsigned int height );

  CameraMode    m_camera_mode;

  ShadeMode     m_shade_mode;
  bool          m_aa_enabled;
  float         m_ao_radius;
  int           m_ao_sample_mult;
  float         m_light_scale;

  Material      m_material;
  Aabb          m_aabb;
  Buffer        m_rnd_seeds;
  Buffer        m_accum_buffer;
  bool          m_accum_enabled;

  Buffer        m_rng_states ;
  bool          m_rng_states_enabled;
  void*         m_dev_rng_states ;  


  float         m_scene_epsilon;
  int           m_frame;
  bool          m_animation;
  char*         m_path ;  
  GGeo*         m_ggeo ;  
};


//------------------------------------------------------------------------------
//
// MeshViewer implementation
//
//------------------------------------------------------------------------------


unsigned int MeshViewer::TOUCH_BAD = 666666u ;


MeshViewer::MeshViewer():
  MeshScene          ( false, false, false ),
  m_camera_mode       ( CM_PINHOLE ),
  m_shade_mode        ( SM_PHONG ),
  m_aa_enabled        ( false ),
  m_ao_radius         ( 1.0f ),
  m_ao_sample_mult    ( 1 ),
  m_light_scale       ( 1.0f ),
  m_accum_enabled     ( false ),
  m_scene_epsilon     ( 1e-4f ),
  m_frame             ( 0 ),
  m_animation         ( false ),
  m_path              ( NULL ),
  m_ggeo              ( NULL ),
  m_rng_states_enabled( false ),
  m_dev_rng_states    ( NULL ) 
{
#if RAYTRACE_CURAND
  m_rng_states_enabled = true ;
#else
  m_rng_states_enabled = false ;
#endif
  printf("MeshViewer::MeshViewer rng_states_enabled RAYTRACE_CURAND  %d \n", m_rng_states_enabled );
  printf("MeshViewer::MeshViewer WIDTH %d HEIGHT %d \n", WIDTH, HEIGHT );
}


GGeo* MeshViewer::getGGeo()
{
    return m_ggeo ; 
}
void MeshViewer::setGGeo(GGeo* ggeo)
{
    m_ggeo = ggeo ; 
}




void MeshViewer::initScene( InitialCameraData& camera_data )
{
  initContext();
  initMaterial();
  initGeometry();
  initLights();   // move lights after geometry, for positioning relative to aabb
  initCamera( camera_data );
  preprocess();

}

optix::Context MeshViewer::getContext()
{
  return m_context ; 
}

void MeshViewer::initContext()
{
  RayTraceConfig* cfg = RayTraceConfig::getInstance(); 
  cfg->Print("MeshViewer::initContext"); 

  bool touch = true ;  

  m_context->setRayTypeCount( 2 );   // initially used touch ray type, but that leads to code duplication
  m_context->setEntryPointCount( touch ? 2 : 1 ); 

  //bool printEnabled = false ; 
  bool printEnabled = true ; 
  m_context->setPrintEnabled(printEnabled); 
  m_context->setPrintBufferSize(8192); 
  m_context->setPrintLaunchIndex(0,0,0);

  //m_context->setStackSize( 1180 );  // original setting
  m_context->setStackSize( 2180 );
  //m_context->setStackSize( 4096 );
  //m_context->setStackSize( 10000 );  // very slow, but succeeds to curand_init with id subsequences

  m_context[ "touch_mode" ]->setUint( 0u );
  m_context[ "radiance_ray_type"   ]->setUint( radiance_ray_type );
  m_context[ "shadow_ray_type"     ]->setUint( shadow_ray_type );

  m_context[ "max_depth"           ]->setInt( 5 );
  m_context[ "ambient_light_color" ]->setFloat( 0.2f, 0.2f, 0.2f );
  m_context[ "output_buffer"       ]->set( createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, WIDTH, HEIGHT) );

  unsigned int elements = WIDTH*HEIGHT ;
  unsigned int optix_device_number = 0u ; 

   // optix managing the rng_states
  if( m_rng_states_enabled )
  { 
      m_rng_states = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_USER);

      m_rng_states->setElementSize(sizeof(curandState));
      m_rng_states->setSize(elements);       // does this destroy prior buffer ?

      m_context[ "rng_states"]->set(m_rng_states);

      m_dev_rng_states = (void*)m_rng_states->getDevicePointer( optix_device_number); 

      bool force = true ;   // forcing as first occasion, ensure start with desired dimension 
      resizeDevRngStates(elements, force);   
  }


  // CUDA managing the rng_states, alternate CUDA-optix interop approach 
  /*
  createDevRngStates( elements ); 
  m_rng_states = m_context->createBufferForCUDA( RT_BUFFER_OUTPUT, RT_FORMAT_USER,  elements );
  m_rng_states->setElementSize(sizeof(curandState));
  m_rng_states->setSize(elements);       // does this destroy prior buffer ?
  m_rng_states->setDevicePointer( optix_device_number, m_dev_rng_states );
  */





  m_context[ "jitter_factor"       ]->setFloat( m_aa_enabled ? 1.0f : 0.0f );
  
  m_accum_enabled = m_aa_enabled                          ||
                    m_shade_mode == SM_AO                 ||
                    m_shade_mode == SM_ONE_BOUNCE_DIFFUSE ||
                    m_shade_mode == SM_AO_PHONG;


  /////  Ray generation program setup

  std::string camera_file ;
  if(m_accum_enabled) camera_file = "accum_camera.cu" ;
  else                camera_file = m_camera_mode == CM_PINHOLE ? "pinhole_camera.cu"  : "orthographic_camera.cu";

  const std::string camera_name = m_camera_mode == CM_PINHOLE ? "pinhole_camera" : "orthographic_camera"; 

  if( m_accum_enabled ) 
  {
      m_accum_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT4, WIDTH, HEIGHT );
      m_context["accum_buffer"]->set( m_accum_buffer ); // keep m_ handle to buffer for resizing 
      resetAccumulation();
  }

  cfg->setRayGenerationProgram(0, camera_file.c_str(), camera_name.c_str() ); 

  cfg->setExceptionProgram(0, camera_file.c_str(), "exception");
  m_context[ "bad_color" ]->setFloat( 0.0f, 1.0f, 0.0f );

  cfg->setMissProgram(0, "constantbg.cu", "miss" ); 
  m_context[ "bg_color" ]->setFloat(  0.34f, 0.55f, 0.85f ); // map(int,np.array([0.34,0.55,0.85])*255) -> [86, 140, 216]



  if(touch)
  {
      m_context["touch_buffer"]->set( m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, 1, 1));
      m_context["touch_bad" ]->setUint( TOUCH_BAD );

      cfg->setRayGenerationProgram(1, camera_file.c_str(), camera_name.c_str() ); 
  }

}



void MeshViewer::createDevRngStates(
      unsigned int elements, 
      unsigned long long seed, 
      unsigned long long offset,
      unsigned int max_blocks,
      unsigned int threads_per_block
    )
{
    LaunchSequence* seq = new LaunchSequence( elements, threads_per_block, max_blocks ) ;
 
    cuRANDWrapper* crw = new cuRANDWrapper(seq, seed, offset);

    crw->setCacheDir(RayTraceConfig::RngDir());

    crw->setCacheEnabled(false);
    //crw->setCacheEnabled(true);

    bool create = true ;    // not creating as using an OptiX managed device Buffer
    crw->Setup(create);  

    delete seq ; 

    m_dev_rng_states = crw->getDevRngStates();

    delete crw ; 
}



void MeshViewer::initDevRngStates(
      unsigned int elements, 
      unsigned long long seed, 
      unsigned long long offset,
      unsigned int max_blocks,
      unsigned int threads_per_block
    )
{
    // hmm need a CUDAConfig object from CUDAWrap to handle 
    // such commonalities as max_blocks and threads_per_block 
    //
    // does this need protection to ensure this only gets run before OptiX launch ?
    //

    printf("MeshViewer::initDevRngStates elements %u seed %llu offset %llu \n", elements, seed, offset );

    LaunchSequence* seq = new LaunchSequence( elements, threads_per_block, max_blocks ) ;

    cuRANDWrapper* crw = new cuRANDWrapper(seq, seed, offset);

    crw->setCacheDir(RayTraceConfig::RngDir());

    crw->setCacheEnabled(false);
    //crw->setCacheEnabled(true);

    crw->setDevRngStates(m_dev_rng_states);

    printf("MeshViewer::initDevRngStates initialize cuRANDWrapper::Setup \n");

    bool create = false ;    // not creating as using an OptiX managed device Buffer
    crw->Setup(create);  

    //
    // loads initial rng_states from cache if available and cache is enabled, 
    // otherwise creates rng_states via CUDA LaunchSequence 
    // and saves to cache
    //
    printf("MeshViewer::initDevRngStates call cuRANDWrapper::Setup DONE  \n");

    delete seq ; 
}

void MeshViewer::resizeDevRngStates(unsigned int elements, bool force)
{
    RTsize size ; 
    m_rng_states->getSize(size);

    if(elements == size && !force)
    {
        printf("MeshViewer::resizeDevRngStates skip as elements unchanged  %u  \n",elements );
    }
    else
    {
        printf("MeshViewer::resizeDevRngStates elements %u START \n",elements );
        m_rng_states->setSize(elements); // assuming this destroys prior buffer 

        unsigned int optix_device_number = 0u ; 
        m_dev_rng_states = (void*)m_rng_states->getDevicePointer( optix_device_number); 

        initDevRngStates(elements);

        printf("MeshViewer::resizeDevRngStates elements %u DONE \n",elements );
    }
}



void MeshViewer::initLights()
{
   // Lights buffer  : pos, color, casts_shadow, padding
   // flipped blue y from -1  to 1 cf g4daeview  (up/down flip somewhere ?) 
   // mapping buffer provides host side pointer to copy to
   // NB BasicLight from commonStructs.h used on both host and device

   float extent = m_aabb.maxExtent() * 1.0 ;
   float light_scale = m_light_scale * 2.0 ; 
   int casts_shadow = 0 ; 
   int padding = 0 ; 

   float3 center = m_aabb.center();

   Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
   light_buffer->setFormat(RT_FORMAT_USER);
   light_buffer->setElementSize(sizeof( BasicLight ) );


   float3 red  = make_float3(1.0f, 0.0f, 0.0f) ;
   float3 green = make_float3(0.0f, 1.0f, 0.0f) ;
   float3 blue = make_float3(0.0f, 0.0f, 1.0f) ;
   float3 white = make_float3(1.0f, 1.0f, 1.0f) ;


   BasicLight lights[] = 
   {
      { center + make_float3( -1.0f, 1.0f,-1.0f )*extent, white*light_scale, casts_shadow, padding },
      { center + make_float3(  1.0f, 1.0f, 1.0f )*extent, white*light_scale, casts_shadow, padding },
      { center + make_float3(  0.0f, 1.0f, 1.0f )*extent, white*light_scale, casts_shadow, padding }
   };

   light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
   memcpy(light_buffer->map(), lights, sizeof(lights));  

   light_buffer->unmap();

   m_context[ "lights" ]->set( light_buffer );
}


void MeshViewer::initMaterial()
{
  RayTraceConfig* cfg = RayTraceConfig::getInstance(); 

  switch( m_shade_mode ) {
    case SM_PHONG: {
      printf("MeshViewer::initMaterial using default SM_PHONG material \n");
      m_material = m_context->createMaterial();
      m_material->setClosestHitProgram( 0, cfg->createProgram("phong.cu", "closest_hit_radiance" ));
      m_material->setAnyHitProgram    ( 1, cfg->createProgram("phong.cu", "any_hit_shadow" ));
      m_material[ "Kd"           ]->setFloat( 0.50f, 0.50f, 0.50f );
      m_material[ "Ks"           ]->setFloat( 0.10f, 0.10f, 0.10f );
      m_material[ "Ka"           ]->setFloat( 0.00f, 0.00f, 0.00f );
      m_material[ "reflectivity" ]->setFloat( 0.10f, 0.10f, 0.10f ); // high reflectivity makes for a confusing image
      m_material[ "phong_exp"    ]->setFloat( 2.00f );
      break;
    }

    case SM_NORMAL: {
      m_material = m_context->createMaterial();
      m_material->setClosestHitProgram( 0, cfg->createProgram( "normal_shader.cu", "closest_hit_radiance" ) );
      break;
    }


    case SM_AO:  break ;
    case SM_ONE_BOUNCE_DIFFUSE: break ;
    case SM_AO_PHONG: break ;


#if 0
    case SM_AO: {
      m_material = m_context->createMaterial();
      m_material->setClosestHitProgram( 0, cfg->createProgram( "ambocc.cu", "closest_hit_radiance" ) );
      m_material->setAnyHitProgram    ( 1, cfg->createProgram( "ambocc.cu", "any_hit_occlusion" ) );    
      break;
    } 
    
    case SM_ONE_BOUNCE_DIFFUSE: {
      m_material = m_context->createMaterial();
      m_material->setClosestHitProgram( 0, cfg->createProgram( "one_bounce_diffuse.cu", "closest_hit_radiance" ) );
      m_material->setAnyHitProgram    ( 1, cfg->createProgram( "one_bounce_diffuse.cu", "any_hit_shadow" ) );
      break;
    }

    case SM_AO_PHONG: {
      m_material = m_context->createMaterial();
      m_material->setClosestHitProgram( 0, cfg->createProgram( "ambocc.cu", "closest_hit_radiance_phong_ao" ) );
      m_material->setAnyHitProgram    ( 1, cfg->createProgram( "ambocc.cu", "any_hit_shadow" ) );
      m_material->setAnyHitProgram    ( 2, cfg->createProgram( "ambocc.cu", "any_hit_occlusion" ) );
      m_context["Kd"]->setFloat(1.0f);
      m_context["Ka"]->setFloat(0.0f);
      m_context["Ks"]->setFloat(0.5f);
      m_context["Kr"]->setFloat(0.0f);
      m_context["phong_exp"]->setFloat(0.0f);
      break;
    }
#endif
  }



  if( m_accum_enabled ) {
    genRndSeeds( WIDTH, HEIGHT );
  }
}


void MeshViewer::initGeometry()
{
  double start, end;
  sutilCurrentTime(&start);

  m_geometry_group = m_context->createGeometryGroup();


  if( G4DAELoader::isMyFile( m_filename ) )
  {
     G4DAELoader loader( m_filename, m_context, m_geometry_group, m_material, m_accel_builder.c_str(), m_accel_traverser.c_str(), m_accel_refine.c_str(), m_accel_large_mesh );
     loader.load();
     setGGeo(loader.getGGeo());
     m_aabb = loader.getSceneBBox();
  } 
  else 
  {
     std::cerr << "Unrecognized model file extension '" << m_filename << "'" << std::endl;
     exit( 0 );
  }

  // Load acceleration structure from a file if that was enabled on the
  // command line, and if we can find a cache file. Note that the type of
  // acceleration used will be overridden by what is found in the file.
  loadAccelCache();

  
  m_context[ "top_object" ]->set( m_geometry_group );
  m_context[ "top_shadower" ]->set( m_geometry_group );

  sutilCurrentTime(&end);
  std::cerr << "Time to load " << (m_accel_large_mesh ? "and cluster " : "") << "geometry: " << end-start << " s.\n";
}


void MeshViewer::initCamera( InitialCameraData& camera_data )
{
  // Set up camera
  float max_dim  = m_aabb.maxExtent();
  float3 eye     = m_aabb.center();
  eye.z         += 2.0f * max_dim;

  camera_data = InitialCameraData( eye,                             // eye
                                   m_aabb.center(),                  // lookat
                                   make_float3( 0.0f, 1.0f, 0.0f ), // up
                                   30.0f );                         // vfov

  // Declare camera variables.  The values do not matter, they will be overwritten in trace.
  m_context[ "eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context[ "U"  ]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context[ "V"  ]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context[ "W"  ]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

}


void MeshViewer::preprocess()
{
  // Settings which rely on previous initialization
  m_scene_epsilon = 1.e-4f * m_aabb.maxExtent();
  m_context[ "scene_epsilon"      ]->setFloat( m_scene_epsilon );
  m_context[ "occlusion_distance" ]->setFloat( m_aabb.maxExtent() * 0.3f * m_ao_radius );

  // Prepare to run 
  m_context->validate();
  double start, end_compile, end_AS_build;
  sutilCurrentTime(&start);
  m_context->compile();
  sutilCurrentTime(&end_compile);
  std::cerr << "Time to compile kernel: "<<end_compile-start<<" s.\n";
  m_context->launch(0,0);
  sutilCurrentTime(&end_AS_build);
  std::cerr << "Time to build AS      : "<<end_AS_build-end_compile<<" s.\n";
}




bool MeshViewer::keyPressed(unsigned char key, int x, int y)
{
   printf("MeshViewer::keyPressed %c x %d y %d \n", key, x, y);
   switch (key)
   {
     case 'e':
       m_scene_epsilon *= .1f;
       std::cerr << "scene_epsilon: " << m_scene_epsilon << std::endl;
       m_context[ "scene_epsilon" ]->setFloat( m_scene_epsilon );
       return true;
     case 'E':
       m_scene_epsilon *= 10.0f;
       std::cerr << "scene_epsilon: " << m_scene_epsilon << std::endl;
       m_context[ "scene_epsilon" ]->setFloat( m_scene_epsilon );
       return true;
     case 'z':
       touch(key, x, y);
       return true ;
   }
   return false;
}

          
void MeshViewer::doResize( unsigned int width, unsigned int height )
{
  printf("MeshViewer::doResize width %d height %d \n", width, height);

  // output_buffer resizing handled in base class
  if( m_accum_enabled ) 
  {
      m_accum_buffer->setSize( width, height );
      m_rnd_seeds->setSize( width, height );
      genRndSeeds( width, height );
      resetAccumulation();
  }

  if( m_rng_states_enabled )
  {
      resizeDevRngStates(width*height);
  }  

}


void MeshViewer::trace( const RayGenCameraData& camera_data )
{
  if (m_animation && GLUTDisplay::isBenchmark() ) 
  {
      static float angleU = 0.0f, angleV = 0.0f, scale = 1.0f, dscale = 0.96f, backside = 0.0f;
      static int phase = 0, accumed_frames = 0;
      const float maxang = M_PIf * 0.2f;
      const float rotvel = M_2_PIf*0.1f;
      float3 c = m_aabb.center();
      float3 e = camera_data.eye;

      Matrix3x3 m = make_matrix3x3(Matrix4x4::rotate(angleV + backside, normalize(camera_data.V)) * 
      Matrix4x4::rotate(angleU, normalize(camera_data.U)) * Matrix4x4::scale(make_float3(scale, scale, scale)));

      if( !m_accum_enabled || accumed_frames++ > 5 ) 
      { // Accumulate 5 frames per animation step.
          switch(phase) 
          {
              case 0: angleV += rotvel; if(angleV > maxang) { angleV =  maxang; phase++; } break;
              case 1: angleU += rotvel; if(angleU > maxang) { angleU =  maxang; phase++; } break;
              case 2: angleV -= rotvel; if(angleV <-maxang) { angleV = -maxang; phase++; } break;
              case 3: angleU -= rotvel; if(angleU <-maxang) { angleU = -maxang; phase=0; } break;
          }

          scale *= dscale;
          if(scale < 0.1f) { dscale = 1.0f / dscale; backside = M_PIf - backside; }
          if(scale > 1.0f) { dscale = 1.0f / dscale; }

          accumed_frames = 0;
          m_camera_changed = true;
      }

      m_context["eye"]->setFloat( c-m*(c-e) );
      m_context["U"]->setFloat( m*camera_data.U );
      m_context["V"]->setFloat( m*camera_data.V );
      m_context["W"]->setFloat( m*camera_data.W );
  } 
  else 
  {
      m_context["eye"]->setFloat( camera_data.eye );
      m_context["U"]->setFloat( camera_data.U );
      m_context["V"]->setFloat( camera_data.V );
      m_context["W"]->setFloat( camera_data.W );
  }

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  if( m_accum_enabled && !m_camera_changed ) 
  {
      // Use more AO samples if the camera is not moving, for increased !/$.
      // Do this above launch to avoid overweighting the first frame
      m_context["sqrt_occlusion_samples"]->setInt( 3 * m_ao_sample_mult );
      m_context["sqrt_diffuse_samples"]->setInt( 3 );
  }

  m_context->launch( 0, static_cast<unsigned int>(buffer_width), static_cast<unsigned int>(buffer_height) );

  if( m_accum_enabled ) 
  {
      ++m_frame; // Update frame number for accumulation.
      if( m_camera_changed ) 
      {
          m_camera_changed = false;
          resetAccumulation();
      }
      // The frame number is used as part of the random seed.
      m_context["frame"]->setInt( m_frame );
  }
}



void MeshViewer::dumpCamera(const char* msg, unsigned char key, int x, int y )
{
    // see $OPTIX_SDK_DIR/sutil/SampleScene.h

    float3 e = m_context["eye"]->getFloat3();
    float3 u = m_context["U"]->getFloat3();
    float3 v = m_context["V"]->getFloat3();
    float3 w = m_context["W"]->getFloat3();

    printf("%s\n", msg );
    printf("eye     %10.3f %10.3f %10.3f    %10.3f  distance to origin  \n", e.x, e.y, e.z, length(e) );
    printf("U ---   %10.3f %10.3f %10.3f    %10.3f  width at focal distance \n", u.x, u.y, u.z, length(u) );
    printf("V  |    %10.3f %10.3f %10.3f    %10.3f  height at focal distance \n", v.x, v.y, v.z, length(v) );
    printf("W  *    %10.3f %10.3f %10.3f    %10.3f  focal distance \n", w.x, w.y, w.z, length(w) );

    Buffer buffer = m_context["output_buffer"]->getBuffer();
    RTsize width, height;
    buffer->getSize( width, height );

    printf("pixel width x height  %lu x %lu  touch %d %d \n", width, height, x, y ); 

}


void MeshViewer::touch(unsigned char key, int x, int y)
{
    /*
       touch_mode launches touch_entry_point

    */


    dumpCamera("MeshViewer::touch", key, x, y );

    Buffer buffer = m_context["output_buffer"]->getBuffer();
    RTsize width, height;
    buffer->getSize( width, height );

    m_context["touch_mode"]->setUint(1u);
    m_context["touch_index"]->setUint(x, height - y ); // by inspection
    m_context["touch_dim"]->setUint(width, height);

    RTsize touch_width = 1u ; 
    RTsize touch_height = 1u ; 
    unsigned int touch_entry_point = 1u ; 

    m_context->launch( touch_entry_point, touch_width, touch_height );

    Buffer touchBuffer = m_context[ "touch_buffer"]->getBuffer();
    m_context["touch_mode"]->setUint(0u);

    unsigned int* touchBuffer_Host = static_cast<unsigned int*>( touchBuffer->map() );
    unsigned int nodeIndex = touchBuffer_Host[0] ;
    touchBuffer->unmap();

    dumpNode(nodeIndex);
}


void MeshViewer::dumpNode(unsigned int nodeIndex)
{
    assert(m_ggeo);
    if(nodeIndex == TOUCH_BAD)
    {
        printf("MeshViewer::dumpNode TOUCH_BAD %u \n", nodeIndex);
    }
    else
    {
        GSolid* solid = m_ggeo->getSolid(nodeIndex) ;
        assert(solid);
        GSubstance* substance = solid->getSubstance();
        printf("MeshViewer::touch nodeIndex %u solid %p  \n", nodeIndex, solid ); 
        substance->dumpTexProps("MeshViewer::touch", 510.f );  // TODO: avoid wavelength duplication
        solid->Summary(NULL); // non-intuitive way to get the detail, fix this
        //substance->Summary("MeshViewer::touch", 20);
    }
}


void MeshViewer::cleanUp()
{
  // Store the acceleration cache if required.
  saveAccelCache();
  SampleScene::cleanUp();
}


Buffer MeshViewer::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}


void MeshViewer::resetAccumulation()
{
    m_frame = 0;
    m_context[ "frame"                  ]->setInt( m_frame );
    m_context[ "sqrt_occlusion_samples" ]->setInt( 1 * m_ao_sample_mult );
    m_context[ "sqrt_diffuse_samples"   ]->setInt( 1 );
}


void MeshViewer::genRndSeeds( unsigned int width, unsigned int height )
{
    // Init random number buffer if necessary.
    if( m_rnd_seeds.get() == 0 ) 
    {
        m_rnd_seeds = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_UNSIGNED_INT, WIDTH, HEIGHT);
        m_context["rnd_seeds"]->setBuffer(m_rnd_seeds);
    }

    // map buffer to provide a host pointer
    unsigned int* seeds = static_cast<unsigned int*>( m_rnd_seeds->map() );
    fillRandBuffer(seeds, width*height);
    m_rnd_seeds->unmap();
}


//-----------------------------------------------------------------------------
//
// Main driver
//
//-----------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << "  -o  | --obj <obj_file>                     Specify .OBJ model to be rendered\n"
    << "  -c  | --cache                              Turn on acceleration structure caching\n"
    << "  -a  | --ao-shade                           Use progressive ambient occlusion shader\n"    
    << "  -ap | --ao-phong-shade                     Use progressive ambient occlusion and phong shader\n"
    << "  -aa | --antialias                          Use subpixel jittering to perform antialiasing\n"
    << "  -n  | --normal-shade                       Use normal shader\n"
    << "  -i  | --diffuse-shade                      Use one bounce diffuse shader\n"
    << "  -O  | --ortho                              Use orthographic camera (cannot use AO mode with ortho)\n"
    << "  -r  | --ao-radius <scale>                  Scale ambient occlusion radius\n"
    << "  -m  | --ao-sample-mult <n>                 Multiplier for the number of AO samples\n"
    << "  -l  | --light-scale <scale>                Scale lights by constant factor\n"
    << "        --large-mesh                         Massive dataset mode\n"
    << "        --animation                          Spin the model (useful for benchmarking)\n"
    << "        --trav <name>                        Acceleration structure traverser\n"
    << "        --build <name>                       Acceleration structure builder\n"
    << "        --refine <n>                         Acceleration structure refinement passes\n"
    << std::endl;
  GLUTDisplay::printUsage();

  std::cerr
    << "App keystrokes:\n"
    << "  e Decrease scene epsilon size (used for shadow ray offset)\n"
    << "  E Increase scene epsilon size (used for shadow ray offset)\n"
    << std::endl;

  if ( doExit ) exit(1);
}



void parseArgs(MeshViewer& scene, GLUTDisplay::contDraw_E& draw_mode, int argc, char** argv)
{
  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "-c" || arg == "--cache" ) {
      scene.setAccelCaching( true );
    } else if( arg == "-n" || arg == "--normal-shade" ) {
      scene.setShadeMode( MeshViewer::SM_NORMAL );
    } else if( arg == "-a" || arg == "--ao-shade" ) {
      scene.setShadeMode( MeshViewer::SM_AO);
      draw_mode = GLUTDisplay::CDProgressive;
    } else if( arg == "-i" || arg == "--diffuse-shade" ) {
      scene.setShadeMode( MeshViewer::SM_ONE_BOUNCE_DIFFUSE );
      draw_mode = GLUTDisplay::CDProgressive;
    } else if( arg == "-ap" || arg == "--ao-phong-shade" ) {
      scene.setShadeMode( MeshViewer::SM_AO_PHONG );
      draw_mode = GLUTDisplay::CDProgressive;
    } else if( arg == "-aa" || arg == "--antialias" ) {
      scene.setAA( true );
      draw_mode = GLUTDisplay::CDProgressive;
    } else if( arg == "-O" || arg == "--ortho" ) {
      scene.setCameraMode( MeshViewer::CM_ORTHO );
    } else if( arg == "-h" || arg == "--help" ) {
      printUsageAndExit( argv[0] ); 
    } else if( arg == "-g" || arg == "--g4dae" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      scene.setMesh(G4DAELoader::identityFilename(argv[++i]));
    } else if( arg == "-o" || arg == "--obj" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      scene.setMesh( argv[++i] );
    } else if( arg == "--trav" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      scene.setTraverser( argv[++i] );
    } else if( arg == "--build" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      scene.setBuilder( argv[++i] );
    } else if( arg == "--refine" ) { // N tree rotation passes to improve BVH quality
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      scene.setRefine( argv[++i] );
    } else if( arg == "--kd" ) {     // Keep this arg for a while for backward compatibility
      scene.setBuilder( "TriangleKdTree" );
      scene.setTraverser( "KdTree" );
    } else if( arg == "--lbvh" ) {   // Keep this arg for a while for backward compatibility
      scene.setBuilder( "Lbvh" );
    } else if( arg == "--bvh" ) {    // Keep this arg for a while for backward compatibility
      scene.setBuilder( "Bvh" );
    } else if( arg == "--large-mesh" ) {
      scene.setLargeMesh( true );
    } else if( arg == "--animation" ) {
      scene.setAnimation( true );
    } else if( arg == "-r" || arg == "--ao-radius" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      scene.setAORadius( static_cast<float>( atof( argv[++i] ) ) );
    } else if( arg == "-m" || arg == "--ao-sample-mult" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      scene.setAOSampleMultiplier( atoi( argv[++i] ) );
    } else if( arg == "-l" || arg == "--light-scale" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      scene.setLightScale( static_cast<float>( atof( argv[++i] ) ) );
    } else {
      std::cerr << "Unknown option: '" << arg << "'" << std::endl;
      printUsageAndExit( argv[0] );
    }
  }
 
  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

}




int main( int argc, char** argv ) 
{

  const std::string target = "MeshViewer" ;

  GLUTDisplay::init( argc, argv );
  GLUTDisplay::contDraw_E draw_mode = GLUTDisplay::CDNone; 

  MeshViewer scene;
  optix::Context context = scene.getContext(); 
  RayTraceConfig* cfg = RayTraceConfig::makeInstance(context, target.c_str());
  scene.setMesh( (std::string( sutilSamplesDir() ) + "/simpleAnimation/cow.obj").c_str() );
  parseArgs(scene, draw_mode, argc, argv );
 
  try 
  {
      GLUTDisplay::run( target, &scene, draw_mode );
  } 
  catch( Exception& e )
  {
      sutilReportError( e.getErrorString().c_str() );
      exit(1);
  }

  return 0;
}
