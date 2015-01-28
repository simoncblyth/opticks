#include "OptiXScene.hh"
#include "OptiXProgram.hh"
#include "OptiXAssimpGeometry.hh"

#include <string.h>
#include <stdlib.h>

#include <sstream>

#include <optixu/optixu_vector_types.h>


OptiXScene::OptiXScene()
        : 
        SampleScene(),
        m_width(1080u),
        m_height(720u),
        m_program(NULL),
        m_geometry(NULL)
{
    printf("OptiXScene ctor\n");
}

OptiXScene::~OptiXScene(void)
{
    printf("OptiXScene dtor\n");
}

optix::Context OptiXScene::getContext()
{
    return m_context ; 
}

void OptiXScene::setProgram(OptiXProgram* program)
{
    m_program = program ; 
}

void OptiXScene::setGeometry(OptiXAssimpGeometry* geometry)
{
    m_geometry = geometry ; 
}


void OptiXScene::setDimensions( const unsigned int w, const unsigned int h ) 
{ 
    m_width = w ; 
    m_height = h ; 
}

void OptiXScene::initScene( InitialCameraData& camera_data )
{
  printf("OptiXScene::initScene  \n");
 
  m_context["max_depth"]->setInt(100);
  m_context["radiance_ray_type"]->setUint(0);
  m_context["shadow_ray_type"]->setUint(1);
  m_context["frame_number"]->setUint( 0u );
  m_context["scene_epsilon"]->setFloat( 1.e-3f );
  m_context["importance_cutoff"]->setFloat( 0.01f );
  m_context["ambient_light_color"]->setFloat( 0.31f, 0.33f, 0.28f );

  m_context["output_buffer"]->set( createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height) );


  const char* filename = "tutorial0.cu" ; 
  optix::Program ray_gen_program = m_program->createProgram(filename, "pinhole_camera" );  
  optix::Program exception_program = m_program->createProgram(filename, "exception" );
  optix::Program miss_program = m_program->createProgram(filename, "miss" );
 
  m_context->setRayGenerationProgram( 0, ray_gen_program ); 
  m_context->setExceptionProgram( 0, exception_program );
  m_context->setMissProgram( 0, miss_program );

  m_context["bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );
  m_context["bg_color"]->setFloat( optix::make_float3( 0.34f, 0.55f, 0.85f ) );
  

 // Set up camera

  //optix::float3 eye  = optix::make_float3( 7.0f, 9.2f, -6.0f ) ;
  //optix::float3 look = optix::make_float3( 0.0f, 4.0f,  0.0f ) ;

  optix::float3 ext = m_geometry->getExtent();
  optix::float3 look = m_geometry->getCenter();
  optix::float3 eye =  look + ext ;

  optix::float3 up   = optix::make_float3( 0.0f, 1.0f,  0.0f ) ; 


  camera_data = InitialCameraData( eye , look, up, 50.0f );

  m_context["eye"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );


  printf("context validate\n");
  m_context->validate();
  printf("context compile\n");
  m_context->compile();
  printf("context compile DONE\n");

}


optix::Buffer OptiXScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}

void OptiXScene::doResize( unsigned int width, unsigned int height )
{
  // output buffer handled in SampleScene::resize
}


void OptiXScene::trace( const RayGenCameraData& camera_data )
{
  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );

  optix::Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  m_context->launch( 0, static_cast<unsigned int>(buffer_width),
                      static_cast<unsigned int>(buffer_height) );
}




