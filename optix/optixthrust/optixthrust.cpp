#include <string>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cuda.h>
#include "optixthrust.hh"

std::string ptxpath(const char* name, const char* cmake_target ){
    char path[128] ; 
    snprintf(path, 128, "%s/%s_generated_%s", getenv("PTXDIR"), cmake_target, name );
    return path ;  
}

//  OptiXThrustMinimal_generated_minimal_float4.cu.ptx


const char* OptiXThrust::CMAKE_TARGET = "OptiXThrustMinimal" ;


OptiXThrust::OptiXThrust(unsigned int buffer_id) :
   m_device(0),
   m_width(100),
   m_height(1),
   m_depth(1),
   m_size(m_width*m_height*m_depth)
{
    m_context = optix::Context::create();
    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    m_context->setStackSize( 2180 );

    if(buffer_id == 0)
    {
        m_buffer = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, m_width, m_height );
        m_context["output_buffer"]->set( m_buffer );
    } 
    else
    {
        m_buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT_OUTPUT, buffer_id);
        m_context["output_buffer"]->set( m_buffer );
    }

    m_context->setEntryPointCount(num_entry);
    addRayGenerationProgram("minimal_float4.cu.ptx", "minimal_float4", raygen_minimal_entry );
    addRayGenerationProgram("minimal_float4.cu.ptx", "dump",           raygen_dump_entry );
}



void OptiXThrust::addRayGenerationProgram( const char* ptxname, const char* progname, unsigned int entry )
{
    std::string path = ptxpath(ptxname, CMAKE_TARGET) ;
    printf("OptiXThrust::addRayGenerationProgram (%d) %s %s : %s \n", entry, ptxname, progname, path.c_str());

    optix::Program prog = m_context->createProgramFromPTXFile( path, progname );
    m_context->setRayGenerationProgram( entry, prog );
}

void OptiXThrust::compile()
{
    m_context->validate();
    m_context->compile();
}

void OptiXThrust::launch(unsigned int entry)
{
    printf("OptiXThrust::launch %d \n", entry);
    m_context->launch(entry, m_width, m_height);
}


