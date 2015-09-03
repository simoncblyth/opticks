#pragma once

#include <optixu/optixpp_namespace.h>

class OptiXThrust {
  public:
       enum { raygen_entry, num_entry } ;
  public:
       OptiXThrust();
       void launch();
  public:
       void postprocess();  // implemented in _postprocess.cu as needs nvcc
  private:
       optix::Context m_context ; 
       optix::Buffer  m_buffer ; 
       unsigned int m_width  ; 
       unsigned int m_height ; 


};
