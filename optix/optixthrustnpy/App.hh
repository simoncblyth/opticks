#pragma once

template <typename T> class NPY ; 
#include <optix_world.h>
#include <optixu/optixpp_namespace.h>

class App {
  public:
     App();
     void loadGenstep();
     void initOptiX();
     void uploadGenstep();
     void checkGenstep();
  public:
     // _.cu
     //template <typename T>
     //void dumpBuffer(const char* msg, optix::Buffer& buffer, unsigned int begin, unsigned int end ); 
  private:
      NPY<float>*    m_gs ; 
      unsigned int   m_gs_size ; 
      optix::Context m_context ; 
      optix::Buffer  m_genstep_buffer ; 
};

inline App::App() : 
     m_gs(NULL), 
     m_gs_size(0) 
{
}

