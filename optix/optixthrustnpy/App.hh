#pragma once

template <typename T> class NPY ; 
#include <optix_world.h>
#include <optixu/optixpp_namespace.h>

class App {
  public:
     App();
     void loadGenstep();
     void initOptiX();
     void uploadEvt();
     void downloadEvt();
     void dumpGensteps();
     void dumpPhotons();
  private:
      NPY<float>*    m_gs_npy ; 
      unsigned int   m_num_gensteps ; 
      unsigned int   m_num_photons ; 
      optix::Context m_context ; 
      optix::Buffer  m_genstep_buffer ; 
      optix::Buffer  m_photon_buffer ; 
};

inline App::App() : 
     m_gs_npy(NULL), 
     m_num_gensteps(0),
     m_num_photons(0) 
{
}

