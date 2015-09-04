#include <string>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include "gloptixthrust.hh"

std::string ptxpath(const char* name, const char* cmake_target ){
    char path[128] ; 
    snprintf(path, 128, "%s/%s_generated_%s", getenv("PTXDIR"), cmake_target, name );
    return path ;  
}

const char* GLOptiXThrust::CMAKE_TARGET = "GLOptiXThrustMinimal" ;


const char* GLOptiXThrust::_OCT  = "OCT  : OptiX-CUDA-Thrust "; 
const char* GLOptiXThrust::_GOCT = "GOCT : OpenGL-OptiX-CUDA-Thrust "; 
const char* GLOptiXThrust::_GCOT = "GCOT : OpenGL-CUDA-OptiX-Thrust "; 
const char* GLOptiXThrust::_GCT  = "GCT  : OpenGL-CUDA-Thrust " ; 


const char* GLOptiXThrust::getInteropDescription()
{
    switch(m_interop)
    {
        case  OCT:return _OCT  ;break;
        case GOCT:return _GOCT ;break;
        case GCOT:return _GCOT ;break;
        case  GCT:return _GCT  ;break;
    }
    return NULL ;
}


void GLOptiXThrust::init()
{
    m_context = optix::Context::create();
    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    m_context->setStackSize( 2180 );

    m_context->setEntryPointCount(num_entry);
    addRayGenerationProgram("circle.cu.ptx", "circle_make_vertices", raygen_minimal_entry );
    addRayGenerationProgram("circle.cu.ptx", "circle_dump",          raygen_dump_entry );


    printf("GLOptiXThrust::init %s \n", getInteropDescription());

}




void GLOptiXThrust::createBuffer()
{
    if( m_buffer_created && m_interop != GCOT )
    {
        printf("GLOptiXThrust::createBuffer skip as exists already and not GCOT \n");
        return ;
    }

    switch(m_interop)
    {
        case GCT:
                  break;
        case OCT:
                  createBufferDefault();
                  break;
        case GOCT:
                  createBufferFromGLBO();
                  break;
        case GCOT:
                  referenceBufferForCUDA();
                  break;
    }
    m_context[m_buffer_name]->set( m_buffer) ; 
    m_buffer_created = true ; 
}

void GLOptiXThrust::cleanupBuffer()
{
    switch(m_interop)
    {
        case GCT:
        case OCT:
        case GOCT:
                  break;
        case GCOT:
                  unreferenceBufferForCUDA();
                  break;
    }
}


void GLOptiXThrust::generate()
{
    createBuffer();
    {
       compile();
       launch(raygen_minimal_entry); // generate vertices
       postprocess(0.1f);  // scale vertices : not reflected in drawing by OpenGL
       launch(raygen_dump_entry);   
    }
    cleanupBuffer();
}

void GLOptiXThrust::update()
{
    createBuffer();
    {
       postprocess(0.1f);  // scale vertices : not reflected in drawing by OpenGL
       sync();
       launch(raygen_dump_entry);   
    }
    cleanupBuffer();

}




void GLOptiXThrust::createBufferFromGLBO()
{
    printf("createBufferFromGLBO %d\n", m_buffer_id);
    m_buffer = m_context->createBufferFromGLBO(m_type, m_buffer_id);
    m_buffer->setFormat( m_format );
    m_buffer->setSize( m_size );
}
void GLOptiXThrust::createBufferDefault()
{
    printf("createBufferDefault\n");
    m_buffer = m_context->createBuffer(m_type, m_format, m_size);
}



void GLOptiXThrust::addRayGenerationProgram( const char* ptxname, const char* progname, unsigned int entry )
{
    std::string path = ptxpath(ptxname, CMAKE_TARGET) ;
    printf("GLOptiXThrust::addRayGenerationProgram (%d) %s %s : %s \n", entry, ptxname, progname, path.c_str());
    optix::Program prog = m_context->createProgramFromPTXFile( path, progname );
    m_context->setRayGenerationProgram( entry, prog );
}

void GLOptiXThrust::markBufferDirty()
{
    printf("markBufferDirty\n");
    m_buffer->markDirty();
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

