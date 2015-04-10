#include "OptiXEngine.hh"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <optixu/optixu.h>
#include <vector>
#include <algorithm>

#include "RayTraceConfig.hh"


#include "stdio.h"

using namespace optix;



enum RayType
{
   radiance_ray_type,
   shadow_ray_type
};



// extracts from /usr/local/env/cuda/OptiX_370b2_sdk/sutil/SampleScene.cpp

OptiXEngine::OptiXEngine() :
    m_use_vbo_buffer(true),
    m_cpu_rendering_enabled(false),
    m_num_devices(0)
{
    printf("OptiXEngine::OptiXEngine\n");
    m_context = Context::create();
}
void OptiXEngine::cleanUp()
{
  m_context->destroy();
  m_context = 0;
}

optix::Context& OptiXEngine::getContext()
{
    return m_context ; 
}

optix::Buffer OptiXEngine::createOutputBuffer(RTformat format, unsigned int width, unsigned int height)
{
  // Set number of devices to be used
  // Default, 0, means not to specify them here, but let OptiX use its default behavior.
  if(m_num_devices)
  {
    int max_num_devices    = Context::getDeviceCount();
    int actual_num_devices = std::min( max_num_devices, std::max( 1, m_num_devices ) );
    std::vector<int> devs(actual_num_devices);
    for( int i = 0; i < actual_num_devices; ++i ) devs[i] = i;
    m_context->setDevices( devs.begin(), devs.end() );
  }

  Buffer buffer;

  if ( m_use_vbo_buffer && !m_cpu_rendering_enabled )
  {
    /*  
      Allocate first the memory for the gl buffer, then attach it to OptiX.
    */
    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    size_t element_size;
    m_context->checkError(rtuGetSizeForRTformat(format, &element_size));
    glBufferData(GL_ARRAY_BUFFER, element_size * width * height, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0); 

    buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, vbo);
    buffer->setFormat(format);
    buffer->setSize( width, height );
  }
  else {
    buffer = m_context->createBuffer( RT_BUFFER_OUTPUT, format, width, height);
  }

  return buffer;
}


void OptiXEngine::initContext(unsigned int width, unsigned int height)
{
    printf("OptiXEngine::initContext\n");

    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    m_context->setPrintLaunchIndex(0,0,0);

    m_context->setStackSize( 2180 );
 
    m_context["output_buffer"]->set( createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, width, height) );


    unsigned int num_entry_points = 1;
    m_context->setEntryPointCount( num_entry_points );

    RayTraceConfig* cfg = RayTraceConfig::getInstance();

    unsigned int entry_point_index = 0 ; 
    cfg->setRayGenerationProgram(entry_point_index, "pinhole_camera.cu", "pinhole_camera" );

    cfg->setExceptionProgram(entry_point_index, "pinhole_camera.cu", "exception");
    m_context[ "bad_color" ]->setFloat( 0.0f, 1.0f, 0.0f );


    m_context[ "radiance_ray_type"   ]->setUint( radiance_ray_type );


    unsigned int num_ray_types = 2 ; 
    m_context->setRayTypeCount( num_ray_types );

    unsigned int ray_type_index = 0 ; 
    cfg->setMissProgram(ray_type_index, "constantbg.cu", "miss" );
    m_context[ "bg_color" ]->setFloat(  0.34f, 0.55f, 0.85f ); // map(int,np.array([0.34,0.55,0.85])*255) -> [86, 140, 216]

}


void OptiXEngine::preprocess()
{
    m_context[ "scene_epsilon"]->setFloat(1.e-4f); //  * m_aabb.maxExtent() );

    m_context->validate();

    m_context->compile();

    m_context->launch(0,0);  // builds Accel Structure

}

void OptiXEngine::trace()
{

   /*
    m_context["eye"]->setFloat( camera_data.eye );
    m_context["U"]->setFloat( camera_data.U );
    m_context["V"]->setFloat( camera_data.V );
    m_context["W"]->setFloat( camera_data.W );
   */

  m_context[ "eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context[ "U"  ]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context[ "V"  ]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context[ "W"  ]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );



    Buffer buffer = m_context["output_buffer"]->getBuffer();
    RTsize buffer_width, buffer_height;
    buffer->getSize( buffer_width, buffer_height );


   m_context->launch( 0, static_cast<unsigned int>(buffer_width), static_cast<unsigned int>(buffer_height) );




}





