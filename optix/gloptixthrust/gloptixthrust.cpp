#include <string>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cuda.h>
#include "gloptixthrust.hh"

std::string ptxpath(const char* name, const char* cmake_target ){
    char path[128] ; 
    snprintf(path, 128, "%s/%s_generated_%s", getenv("PTXDIR"), cmake_target, name );
    return path ;  
}

const char* GLOptiXThrust::CMAKE_TARGET = "GLOptiXThrustMinimal" ;


GLOptiXThrust::GLOptiXThrust(unsigned int buffer_id, unsigned int nvert) :
   m_device(0),
   m_width(nvert),
   m_height(1),
   m_depth(1),
   m_size(m_width*m_height*m_depth)
{
    m_context = optix::Context::create();
    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    m_context->setStackSize( 2180 );

    RTformat format = RT_FORMAT_FLOAT4 ;
    unsigned int size = m_width ; 

    //unsigned int type = RT_BUFFER_OUTPUT ; 
    unsigned int type = RT_BUFFER_INPUT_OUTPUT ; 

    if(buffer_id == 0)
    {
        printf("createBuffer\n");
        m_buffer = m_context->createBuffer(type, format, size);
        m_context["output_buffer"]->set( m_buffer );
    } 
    else
    {
        printf("createBufferFromGLBO\n");
        m_buffer = m_context->createBufferFromGLBO(type, buffer_id);
        m_buffer->setFormat( format );
        m_buffer->setSize( size );
        m_context["output_buffer"]->set( m_buffer );
    }

    m_context->setEntryPointCount(num_entry);
    addRayGenerationProgram("circle.cu.ptx", "circle_make_vertices", raygen_minimal_entry );
    addRayGenerationProgram("circle.cu.ptx", "circle_dump",          raygen_dump_entry );
}

void GLOptiXThrust::addRayGenerationProgram( const char* ptxname, const char* progname, unsigned int entry )
{
    std::string path = ptxpath(ptxname, CMAKE_TARGET) ;
    printf("GLOptiXThrust::addRayGenerationProgram (%d) %s %s : %s \n", entry, ptxname, progname, path.c_str());
    optix::Program prog = m_context->createProgramFromPTXFile( path, progname );
    m_context->setRayGenerationProgram( entry, prog );
}

void GLOptiXThrust::compile()
{
    m_context->validate();
    m_context->compile();
}

void GLOptiXThrust::launch(unsigned int entry)
{
    printf("GLOptiXThrust::launch %d \n", entry);
    m_context->launch(entry, m_width, m_height);
}

