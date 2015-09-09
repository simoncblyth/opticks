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

const char* OptiXThrust::CMAKE_TARGET = "OptiXThrustMinimal" ;


void OptiXThrust::init()
{
    m_context = optix::Context::create();
    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    m_context->setStackSize( 2180 );

    m_buffer = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, m_size );
    m_context["output_buffer"]->set( m_buffer );

    m_context->setEntryPointCount(num_entry);

    addRayGenerationProgram("minimal_float4.cu.ptx", "minimal", raygen_minimal_entry );
    addRayGenerationProgram("minimal_float4.cu.ptx", "circle",  raygen_circle_entry );
    addRayGenerationProgram("minimal_float4.cu.ptx", "dump",    raygen_dump_entry );
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
    m_context->launch(entry, m_size );
}


