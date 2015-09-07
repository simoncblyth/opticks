#pragma once

#include <optixu/optixpp_namespace.h>

/*

Access needed by OpenGL/OptiX/Thrust for each buffer,
dictates the Interop approach taken.  

** Despite many attempts reliable 3-way interop just doesnt work, 
   so arrange communications be be pairwise**

For example in ggeoview the photon data "vtx" and "seq"
are written by OptiX. The "seq" needs to be read by 
Thrust to derive the "sel" (via an indexing process). 

            OpenGL  OptiX  Thrust 
  vtx         R       W       -
  seq         -       W       R
  sel         R       -       W 

*/


struct Resource ; 

class OBuffer {
  public:
       typedef enum { RW, R, W } Access_t ; 
       OBuffer(optix::Context& context, unsigned int buffer_id, const char* buffer_name, unsigned int count, Access_t access=OBuffer::RW );
  private:
       void init();
  public:
       void mapGLToCUDA();
       void unmapGLToCUDA();
  public:
       void mapGLToCUDAToOptiX();
       void unmapGLToCUDAToOptiX();
  public:
       void mapGLToOptiX();
       void unmapGLToOptiX();
  public:
       void mapOptiXToCUDA();
       void unmapOptiXToCUDA();
  public:
       void streamSync();
       void* getRawPointer();
       unsigned int getSize();
       bool  isMapped(); 
  private:
       optix::Context        m_context ; 
       optix::Buffer         m_buffer ; 
       Resource*             m_resource ; 
       unsigned int          m_device ; 
       unsigned int          m_buffer_id ;
       const char*           m_buffer_name ;
       Access_t              m_access ; 
       unsigned int          m_width  ; 
       unsigned int          m_height ; 
       unsigned int          m_depth ; 
       unsigned int          m_size ; 
       RTformat              m_format ; 
       unsigned int          m_type ; 
       void*                 m_dptr ;
       bool                  m_mapped ; 

};


inline OBuffer::OBuffer(optix::Context& context, unsigned int buffer_id, const char* buffer_name, unsigned int count, Access_t access ) :
   m_context(context),
   m_resource(NULL),
   m_device(0),
   m_buffer_id(buffer_id),
   m_buffer_name(strdup(buffer_name)),
   m_access(access),
   m_width(count),
   m_height(1),
   m_depth(1),
   m_size(m_width*m_height*m_depth),
   m_format(RT_FORMAT_FLOAT4),
   m_type(RT_BUFFER_INPUT_OUTPUT),
   m_dptr(NULL),
   m_mapped(false)
{
   init();
}



inline void* OBuffer::getRawPointer()
{
   return m_dptr ;
}
inline bool OBuffer::isMapped()
{
   return m_mapped ; 
}
inline unsigned int OBuffer::getSize()
{
   return m_size ; 
}
