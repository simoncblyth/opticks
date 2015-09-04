#pragma once

#include <optixu/optixpp_namespace.h>

//template <typename T> class CudaGLBuffer ;

class GLOptiXThrust {
  public:
       static const char* _OCT ; 
       static const char* _GOCT ; 
       static const char* _GCOT ; 
       static const char* _GCT ; 

       static const char* CMAKE_TARGET ; 
       enum { raygen_minimal_entry, raygen_dump_entry, num_entry } ;
       typedef enum { OCT, GOCT, GCOT, GCT } Interop_t ;  
  public:
       GLOptiXThrust(unsigned int buffer_id, const char* buffer_name, unsigned int nvert, Interop_t interop);
       const char* getInteropDescription();
  private:
       void init();
  public:
       void generate();
       void update();
  public:
       void createBuffer();
       void cleanupBuffer();
  private:
       void createBufferDefault();
       void createBufferFromGLBO();
       void referenceBufferForCUDA();
  private:
       void unreferenceBufferForCUDA();
  private:
       void markBufferDirty();
       void addRayGenerationProgram( const char* ptxname, const char* progname, unsigned int entry );
  public:
       void compile();
       void launch(unsigned int entry);
  public:
       // methods implemented in _postprocess.cu as needs nvcc
       void postprocess(float factor); 
       void sync(); 
       template <typename T> T* getRawPointer(Interop_t interop);
       template <typename T> T* getRawPointer();
  private:
       unsigned int          m_device ; 
       unsigned int          m_buffer_id ;
       const char*           m_buffer_name ;
       bool                  m_buffer_created ; 
       Interop_t             m_interop ; 
       optix::Context        m_context ; 
       optix::Buffer         m_buffer ; 
       unsigned int          m_width  ; 
       unsigned int          m_height ; 
       unsigned int          m_depth ; 
       unsigned int          m_size ; 
       RTformat              m_format ; 
       unsigned int          m_type ; 

};


inline GLOptiXThrust::GLOptiXThrust(unsigned int buffer_id, const char* buffer_name, unsigned int nvert, Interop_t interop ) :
   m_device(0),
   m_buffer_id(buffer_id),
   m_buffer_name(strdup(buffer_name)),
   m_buffer_created(false),
   m_interop(interop),
   m_width(nvert),
   m_height(1),
   m_depth(1),
   m_size(m_width*m_height*m_depth),
   m_format(RT_FORMAT_FLOAT4),
   m_type(RT_BUFFER_INPUT_OUTPUT)
{
   init();
}

