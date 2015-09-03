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


OptiXThrust::OptiXThrust() :
   m_width(100),
   m_height(1)
{
    m_context = optix::Context::create();
    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    m_context->setStackSize( 2180 );
    m_context->setEntryPointCount(num_entry);

    m_buffer = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, m_width, m_height );
    m_context["output_buffer"]->set( m_buffer );

    std::string path = ptxpath("minimal_float4.cu.ptx", "OptiXThrustMinimal") ;
    printf("OptiXThrust::OptiXThrust %s \n", path.c_str());

    optix::Program raygen = m_context->createProgramFromPTXFile( path, "minimal_float4" );
    m_context->setRayGenerationProgram( raygen_entry, raygen );
}


void OptiXThrust::launch()
{
    m_context->validate();
    m_context->compile();
    m_context->launch(0,0);
    m_context->launch(raygen_entry, m_width, m_height);
}


