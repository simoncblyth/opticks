#include "OptiXEngine.hh"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <optixu/optixu.h>
#include <vector>
#include <algorithm>

#include "RayTraceConfig.hh"

#include "assert.h"
#include "stdio.h"

using namespace optix;



enum RayType
{
   radiance_ray_type,
   shadow_ray_type
};



// extracts from /usr/local/env/cuda/OptiX_370b2_sdk/sutil/SampleScene.cpp

OptiXEngine::OptiXEngine() :
    m_width(0),
    m_height(0),
    m_vbo(0),
    m_pbo(0),
    m_vbo_element_size(0),
    m_pbo_element_size(0)
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

void OptiXEngine::setSize(unsigned int width, unsigned int height)
{
    m_width = width ;
    m_height = height ;
}


optix::Buffer OptiXEngine::createOutputBuffer(RTformat format, unsigned int width, unsigned int height)
{
    Buffer buffer;
    buffer = m_context->createBuffer( RT_BUFFER_OUTPUT, format, width, height);
    return buffer ; 
}
optix::Buffer OptiXEngine::createOutputBuffer_VBO(RTformat format, unsigned int width, unsigned int height)
{
    Buffer buffer;

    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    m_context->checkError(rtuGetSizeForRTformat(format, &m_vbo_element_size));
    glBufferData(GL_ARRAY_BUFFER, m_vbo_element_size * width * height, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0); 

    buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, m_vbo);
    buffer->setFormat(format);
    buffer->setSize( width, height );

    return buffer;
}
optix::Buffer OptiXEngine::createOutputBuffer_PBO(RTformat format, unsigned int width, unsigned int height)
{
    Buffer buffer;

    glGenBuffers(1, &m_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);

    m_context->checkError(rtuGetSizeForRTformat(format, &m_pbo_element_size));
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_pbo_element_size * width * height, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); 

    buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, m_pbo);
    buffer->setFormat(format);
    buffer->setSize( width, height );

    printf("OptiXEngine::createOutputBuffer_PBO  m_pbo_element_size %lu width %u height %u m_pbo %u \n", m_pbo_element_size, width, height, m_pbo );  
    //fill_PBO(); // dummy 
  
    return buffer;
}

void OptiXEngine::fill_PBO()
{
    //  https://www.opengl.org/wiki/Pixel_Buffer_Object
    //  https://www.opengl.org/wiki/Pixel_Transfer

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    void* pboData = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);

    for(unsigned int w=0 ; w<m_width ; ++w ){
    for(unsigned int h=0 ; h<m_height ; ++h ) 
    {
        unsigned char* p = (unsigned char*)pboData ; 
        *(p+0) = 0xAA ;
        *(p+1) = 0xBB ;
        *(p+2) = 0xCC ;
        *(p+3) = 0x00 ;
    }
    } 
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER); 
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}


void OptiXEngine::associate_PBO_to_Texture(unsigned int texId)
{
    assert(m_pbo > 0);
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBindTexture( GL_TEXTURE_2D, texId );

    // this kills the teapot
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_BGRA, GL_UNSIGNED_BYTE, NULL );
}



void OptiXEngine::initContext(unsigned int width, unsigned int height)
{
    setSize(width, height);

    printf("OptiXEngine::initContext\n");

    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    m_context->setPrintLaunchIndex(0,0,0);

    m_context->setStackSize( 2180 );
 
    m_context["output_buffer"]->set( createOutputBuffer_PBO(RT_FORMAT_UNSIGNED_BYTE4, width, height) );

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





