#include <GL/glew.h>
#include <GLFW/glfw3.h>

// npy-
#include "NLog.hpp"

// oglrap-
#include "Texture.hh"

// optixrap-
#include "OFrame.hh"
#include "OContext.hh"

#include <optixu/optixu.h>
//#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>
#include <cassert>

using namespace optix ; 


void OFrame::init(unsigned int width, unsigned int height)
{
    m_width = width ; 
    m_height = height ; 

    // generates the m_pbo and m_depth identifiers and buffers
    m_output_buffer = createOutputBuffer_PBO(m_pbo, RT_FORMAT_UNSIGNED_BYTE4, width, height) ;


    m_texture = new Texture();   // QuadTexture would be better name
    m_texture->setSize(width, height);
    m_texture->create();

    m_texture_id = m_texture->getId() ;

    LOG(debug) << "OFrame::init size(" << width << "," << height << ")  texture_id " << m_texture_id ;


    m_touch_buffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT4, 1, 1);

    m_context["touch_buffer"]->set( m_touch_buffer );
    m_context["touch_mode" ]->setUint( 0u );
}


void OFrame::setSize(unsigned int width, unsigned int height)
{
    assert(0);
}



///usr/local/env/cuda/OptiX_370b2_sdk/sutil/MeshScene.cpp

// "touch" mode is tied to the active rendering (currently only e_pinhole_camera)
// as the meaning of x,y mouse/trackpad touches depends on that rendering.  
// Because of this using a separate "touch" entry point may not so useful ?
// Try instead splitting at ray type level.
//
// But the output requirements are very different ? Which would argue for a separate entry point.


optix::Buffer OFrame::createOutputBuffer_PBO(unsigned int& id, RTformat format, unsigned int width, unsigned int height, bool depth)
{
    Buffer buffer;

    glGenBuffers(1, &id);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, id);

    size_t element_size ; 
    m_context->checkError(rtuGetSizeForRTformat(format, &element_size));

    LOG(info) << "OFrame::createOutputBuffer_PBO" 
              <<  " element_size " << element_size 
              ;

    assert(element_size == 4);

    unsigned int nbytes = element_size * width * height ;

    m_pbo_data = (unsigned char*)malloc(nbytes);
    memset(m_pbo_data, 0x88, nbytes);  // initialize PBO to grey 

    glBufferData(GL_PIXEL_UNPACK_BUFFER, nbytes, m_pbo_data, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); 

    buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, id);
    buffer->setFormat(format);
    buffer->setSize( width, height );

    LOG(debug) << "OFrame::createOutputBuffer_PBO  element_size " << element_size << " size (" << width << "," << height << ") pbo id " << id ;
  
    return buffer;
}


void OFrame::push_PBO_to_Texture()
{
    m_push_count += 1 ; 
    push_Buffer_to_Texture( m_output_buffer, m_pbo, m_texture_id, false );    
}


void OFrame::push_Buffer_to_Texture(optix::Buffer& buffer, int buffer_id, int texture_id, bool depth)
{
    RTsize buffer_width_rts, buffer_height_rts;
    buffer->getSize( buffer_width_rts, buffer_height_rts );

    int buffer_width  = static_cast<int>(buffer_width_rts);
    int buffer_height = static_cast<int>(buffer_height_rts);

    RTformat buffer_format = buffer->getFormat();

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
    // send pbo data to the texture

    assert(buffer_id > 0);

    glBindTexture(GL_TEXTURE_2D, texture_id );

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer_id);

    RTsize elementSize = buffer->getElementSize();
    if      ((elementSize % 8) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if ((elementSize % 4) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if ((elementSize % 2) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else                             glPixelStorei(GL_UNPACK_ALIGNMENT, 1);


    GLenum target = GL_TEXTURE_2D ;
    GLint level = 0 ;            // level-of-detail number. Level 0 is the base image level
    GLint internalFormat ;       // number of color components in the texture (1, 2, 3, or 4), or a symbolic constant like: GL_RGBA8 
    GLint border = 0 ; 

    // pixel data : format, type, data (offset)
    // 
    //    if a non-zero named buffer object is bound to the GL_PIXEL_UNPACK_BUFFER
    //    target (see glBindBuffer) while a texture image is specified, data is treated
    //    as a byte offset into the buffer object's data store. 
    //
    GLenum format ;
    GLenum type ;
    const GLvoid * data = 0 ;  

    switch(buffer_format) 
    {
        case RT_FORMAT_UNSIGNED_BYTE4:
            internalFormat = GL_RGBA8 ;  
            format = GL_BGRA ;
            type = GL_UNSIGNED_BYTE ;
            break ; 
        case RT_FORMAT_FLOAT4:
            internalFormat = GL_RGBA32F_ARB ;
            format = GL_RGBA ;
            type = GL_FLOAT ; 
            break;
        case RT_FORMAT_FLOAT3:
            internalFormat = GL_RGBA32F_ARB ;
            format = GL_RGB ;
            type = GL_FLOAT ; 
            break;
        case RT_FORMAT_FLOAT:
            internalFormat = GL_LUMINANCE32F_ARB ;
            format = GL_LUMINANCE ;
            type = GL_FLOAT ; 
            break;
        default:
            assert(0 && "Unknown buffer format");
    }
  

    // (optix-pdf interop chapter):
    //
    //     Not all OpenGL texture formats are supported by OptiX. A table that lists the
    //     supported texture formats can be found in Appendix A.
    //
    //     They include
    //           GL_RGBA8
    //           GL_R32F 
    //

    if(depth)
    {
      // guessing
        internalFormat = GL_R32F ;
        format = GL_DEPTH_COMPONENT ;  // means will be clamped into 0,1 (and may be scaled, biased with GL_DEPTH_SCALE, GL_DEPTH_BIAS)
        type = GL_FLOAT ;
    }

    glTexImage2D(target, level, internalFormat, buffer_width, buffer_height, border, format, type, data);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    //glBindTexture(GL_TEXTURE_2D, 0 );   get blank screen when do this here

}




optix::Buffer OFrame::createOutputBuffer_VBO(unsigned int& id, RTformat format, unsigned int width, unsigned int height)
{
    Buffer buffer;

    glGenBuffers(1, &id);
    glBindBuffer(GL_ARRAY_BUFFER, id);

    size_t element_size ; 
    m_context->checkError(rtuGetSizeForRTformat(format, &element_size));
    assert(element_size == 16);

    const GLvoid *data = NULL ;
    glBufferData(GL_ARRAY_BUFFER, element_size * width * height, data, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0); 

    buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, id);
    buffer->setFormat(format);
    buffer->setSize( width, height );

    return buffer;
}



optix::Buffer OFrame::createOutputBuffer(RTformat format, unsigned int width, unsigned int height)
{
    Buffer buffer;
    buffer = m_context->createBuffer( RT_BUFFER_OUTPUT, format, width, height);
    return buffer ; 
}




void OFrame::fill_PBO()
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




// fulfil Touchable interface
unsigned int OFrame::touch(int ix_, int iy_)
{
    assert(0);
    if(m_push_count == 0)
    {
        LOG(warning) << "OFrame::touch \"OptiX touch mode\" only works after performing an OptiX trace, press O to toggle OptiX tracing then try again " ; 
        return 0 ; 
    }


    // (ix_, iy_) 
    //        (0,0)              at top left,  
    //   (1024,768)*pixel_factor at bottom right


    RTsize width, height;
    m_output_buffer->getSize( width, height );

    int ix = ix_ ; 
    int iy = height - iy_;   

    // (ix,iy) 
    //   (0,0)                     at bottom left
    //   (1024,768)*pixel_factor   at top right  

    m_context["touch_mode"]->setUint(1u);
    m_context["touch_index"]->setUint(ix, iy ); // by inspection
    m_context["touch_dim"]->setUint(width, height);

    //RTsize touch_width = 1u ; 
    //RTsize touch_height = 1u ; 

    // TODO: generalize touch to work with the active camera (eg could be orthographic)
    assert(0); // this needs a rethink as no longer using the entry enum
    // m_context->launch( OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH|OContext::LAUNCH, OContext::e_pinhole_camera_entry , touch_width, touch_height );

    Buffer touchBuffer = m_context[ "touch_buffer"]->getBuffer();
    m_context["touch_mode"]->setUint(0u);

    uint4* touchBuffer_Host = static_cast<uint4*>( touchBuffer->map() );
    uint4 touch = touchBuffer_Host[0] ;
    touchBuffer->unmap();

    LOG(info) << "OFrame::touch "
              << " ix_ " << ix_ 
              << " iy_ " << iy_   
              << " ix " << ix 
              << " iy " << iy   
              << " width " << width   
              << " height " << height 
              << " touch.x nodeIndex " << touch.x 
              << " touch.y " << touch.y 
              << " touch.z " << touch.z   
              << " touch.w " << touch.w 
              ;  

     unsigned int target = touch.x ; 

    return target ; 
}


