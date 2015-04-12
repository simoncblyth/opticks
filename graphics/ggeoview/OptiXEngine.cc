#include "OptiXEngine.hh"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <optixu/optixu.h>
#include <vector>
#include <algorithm>

#include "Composition.hh"
#include "Renderer.hh"
#include "Texture.hh"
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

OptiXEngine::OptiXEngine(const char* cmake_target) :
    m_vbo(0),
    m_pbo(0),
    m_vbo_element_size(0),
    m_pbo_element_size(0),
    m_pbo_data(NULL),
    m_composition(NULL),
    m_renderer(NULL),
    m_texture(NULL),
    m_config(NULL)
{
    printf("OptiXEngine::OptiXEngine\n");

    m_context = Context::create();
    m_geometry_group = m_context->createGeometryGroup();

    m_config = RayTraceConfig::makeInstance(m_context, cmake_target);


    // TODO: geometry loading
    //m_context[ "top_object" ]->set( m_geometry_group );

    m_renderer = new Renderer();
    m_texture = new Texture();

    printf("OptiXEngine::OptiXEngine DONE\n");
}

void OptiXEngine::setComposition(Composition* composition)
{
    assert(m_renderer);
    m_composition = composition ; 
    m_renderer->setComposition(composition);
}


void OptiXEngine::initContext()
{
    unsigned int width  = m_composition->getWidth();
    unsigned int height = m_composition->getHeight();

    printf("OptiXEngine::initContext %u %u\n", width, height);

    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    m_context->setPrintLaunchIndex(0,0,0);

    m_context->setStackSize( 2180 );
 
    m_output_buffer = createOutputBuffer_PBO(RT_FORMAT_UNSIGNED_BYTE4, width, height) ;
    m_context["output_buffer"]->set( m_output_buffer );

    m_context["touch_buffer"]->set( m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, 1, 1));


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

    m_composition->setSize(width, height);
    m_texture->setSize(width, height);
    
}



/*
124 void SampleScene::resize(unsigned int width, unsigned int height)
125 {
126   try {
127     Buffer buffer = getOutputBuffer();
128     buffer->setSize( width, height );
129 
130     if(m_use_vbo_buffer)
131     {
132       buffer->unregisterGLBuffer();
133       glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer->getGLBOId());
134       glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer->getElementSize() * width * height, 0, GL_STREAM_DRAW);
135       glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
136       buffer->registerGLBuffer();
137     }

*/





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

    assert(m_pbo_element_size == 4);
    unsigned int nbytes = m_pbo_element_size * width * height ;

    m_pbo_data = (unsigned char*)malloc(nbytes);
    memset(m_pbo_data, 0x88, nbytes);  // initialize PBO to grey 

    glBufferData(GL_PIXEL_UNPACK_BUFFER, nbytes, m_pbo_data, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); 

    buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, m_pbo);
    buffer->setFormat(format);
    buffer->setSize( width, height );

    printf("OptiXEngine::createOutputBuffer_PBO  m_pbo_element_size %lu width %u height %u m_pbo %u \n", m_pbo_element_size, width, height, m_pbo );  
    //fill_PBO(); // dummy 
  
    return buffer;
}



void OptiXEngine::associate_PBO_to_Texture(unsigned int texId)
{
    assert(m_pbo > 0);
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBindTexture( GL_TEXTURE_2D, texId );

    // this kills the teapot
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_BGRA, GL_UNSIGNED_BYTE, NULL );
}



void OptiXEngine::displayFrame(unsigned int texId)
{
   // see  GLUTDisplay::displayFrame() 

    RTsize buffer_width_rts, buffer_height_rts;
    m_output_buffer->getSize( buffer_width_rts, buffer_height_rts );

    int buffer_width  = static_cast<int>(buffer_width_rts);
    int buffer_height = static_cast<int>(buffer_height_rts);

    RTformat buffer_format = m_output_buffer->getFormat();

    //
    // glTexImage2D specifies mutable texture storage characteristics and provides the data
    //
    //    *internalFormat* 
    //         format with which OpenGL should store the texels in the texture
    //    *data*
    //         location of the initial texel data in host memory, 
    //         if a buffer is bound to the GL_PIXEL_UNPACK_BUFFER binding point, 
    //         texel data is read from that buffer object, and *data* is interpreted 
    //         as an offset into that buffer object from which to read the data. 
    //    *format* and *type*
    //         initial source texel data layout which OpenGL will convert 
    //         to the internalFormat
    // 

   // send pbo to texture

    assert(m_pbo > 0);

    glBindTexture(GL_TEXTURE_2D, texId );

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);

    RTsize elementSize = m_output_buffer->getElementSize();
    if      ((elementSize % 8) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if ((elementSize % 4) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if ((elementSize % 2) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else                             glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    switch(buffer_format) 
    {   //               target   miplevl  internalFormat                     border  format   type           data  
        case RT_FORMAT_UNSIGNED_BYTE4:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, buffer_width, buffer_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);
            break ; 
        case RT_FORMAT_FLOAT4:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, buffer_width, buffer_height, 0, GL_RGBA, GL_FLOAT, 0);
            break;
        case RT_FORMAT_FLOAT3:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, buffer_width, buffer_height, 0, GL_RGB, GL_FLOAT, 0);
            break;
        case RT_FORMAT_FLOAT:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, buffer_width, buffer_height, 0, GL_LUMINANCE, GL_FLOAT, 0);
            break;
        default:
            assert(0 && "Unknown buffer format");
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


}








void OptiXEngine::fill_PBO()
{
    // not working
    //
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





