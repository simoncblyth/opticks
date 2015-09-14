#pragma once

#include <optixu/optixpp_namespace.h>

// NB : THIS IS DEMO/TESTING CODE **NOT REUSABLE CODE**

class OptiXThrust {
  public:
       enum { raygen_minimal_entry, raygen_circle_entry, raygen_dump_entry, num_entry } ;
  public:
       OptiXThrust(unsigned int size);
  private:
       void init();
  public:
       void addRayGenerationProgram( const char* ptxname, const char* progname, unsigned int entry );
       void compile();
  public:
       void minimal();
       void circle();
       void dump();
  private:
       void launch(unsigned int entry);
  public:
       // implemented in _.cu for nvcc compilaiton
       void photon_test();
       void postprocess(); 
       void compaction(); 
       void compaction4(); 
       void strided(); 
       void strided4(); 
       void for_each_dump(); 
       void sync(); 

       optix::float4* make_masked_buffer( int* d_mask, unsigned int mask_size, unsigned int num );
       void dump_photons( optix::float4* host , unsigned int num );

  private:
       unsigned int m_device ; 
       optix::Context m_context ; 
       optix::Buffer  m_buffer ; 
       unsigned int m_size ; 
};


inline OptiXThrust::OptiXThrust(unsigned int size) :
   m_device(0),
   m_size(size)
{
    init();
}

inline void OptiXThrust::minimal()
{
    launch(raygen_minimal_entry);
}
inline void OptiXThrust::circle()
{
    launch(raygen_circle_entry);
}
inline void OptiXThrust::dump()
{
    launch(raygen_dump_entry);
}




