#include <string>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <optixu/optixpp_namespace.h>

std::string ptxpath(const char* name){
    char path[128] ; 
    snprintf(path, 128, "%s/%s", getenv("PTXDIR"), name );
    return path ;  
}

enum { raygen_entry, num_entry } ;
unsigned int width  = 512 ; 
unsigned int height = 512 ; 

int main( int argc, char** argv )
{
  
    optix::Context m_context = optix::Context::create();
    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    m_context->setStackSize( 2180 );
    m_context->setEntryPointCount(num_entry);

    // for OpenGL interop, need to createBufferFromGLBO
    optix::Buffer m_output_buffer = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height );
    m_context["output_buffer"]->set( m_output_buffer );

    optix::Program raygen = m_context->createProgramFromPTXFile( ptxpath("minimal.ptx"), "minimal" );
    m_context->setRayGenerationProgram( raygen_entry, raygen );

    m_context->validate();
    m_context->compile();
    m_context->launch(0,0);
    m_context->launch(raygen_entry, width, height);


    unsigned char* ptr = (unsigned char*)m_output_buffer->map();
    for(unsigned int i=0 ; i < width*height*4 ; i++){
        unsigned char v = *(ptr + i );
        assert( v == 128 );
        //std::cout << int(v) << ( i % width == 0 ? "\n" : " " ) ; 
    }
    m_output_buffer->unmap();

    return 0 ; 
}

