#pragma once
#include <optixu/optixpp_namespace.h>

class GLOptiXThrust {
  public:
       static const char* CMAKE_TARGET ; 
       enum { raygen_minimal_entry, raygen_dump_entry, num_entry } ;
  public:
       GLOptiXThrust(unsigned int buffer_id, unsigned int nvert);
       void addRayGenerationProgram( const char* ptxname, const char* progname, unsigned int entry );
       void compile();
  public:
       void launch(unsigned int entry);
  public:
       // methods implemented in _postprocess.cu as needs nvcc
       void postprocess(float factor); 
       void sync(); 
  private:
       unsigned int m_device ; 
       optix::Context m_context ; 
       optix::Buffer  m_buffer ; 
       unsigned int m_width  ; 
       unsigned int m_height ; 
       unsigned int m_depth ; 
       unsigned int m_size ; 

};
