#pragma once

#include <optixu/optixpp_namespace.h>
#include "CResource.hh" 
#include "BufSpec.hh"

class OBuffer {
  public:
      typedef enum { UNMAPPED, GLToCUDA, GLToCUDAToOptiX, GLToOptiX, OptiXToCUDA, OptiX } Mapping_t ;   

      static const char* UNMAPPED_ ;
      static const char* GLToCUDA_;
      static const char* GLToCUDAToOptiX_;
      static const char* GLToOptiX_;
      static const char* OptiXToCUDA_;
      static const char* OptiX_;

  public:
       OBuffer(optix::Context& context, 
               unsigned int buffer_id,              /* OpenGL id, 0:indicates no OpenGL backing */
               const char* buffer_name,             /* OptiX context variable name */
               unsigned int count,                  /* number of typed entries */
               RTformat format,                     /* OptiX buffer format */
               unsigned int type,                   /* OptiX buffer access pattern */
               CResource::Access_t access = CResource::RW );
  private:
       void init();
       void preqs();
  public:
        BufSpec map(OBuffer::Mapping_t mapping); 
        void unmap();
  private:
       void mapGLToCUDA();
       void unmapGLToCUDA();
       void mapGLToCUDAToOptiX();
       void unmapGLToCUDAToOptiX();
       void mapGLToOptiX();
       void unmapGLToOptiX();
       void mapOptiXToCUDA();
       void unmapOptiXToCUDA();
       void mapOptiX();
       void unmapOptiX();
  private:
       const char* getMappingDescription();
       void streamSync();
       //void* getRawPointer();
       //unsigned int getSize();
       //unsigned int getNumBytes();
       //BufSpec getBufSpec();
       bool  isMapped(); 
  private:
       unsigned int getBufferSize();
       unsigned int getElementSize();
       void fillBufSpec(void* dev_ptr);
  private:
       optix::Context        m_context ; 
       optix::Buffer         m_buffer ; 
       CResource*            m_resource ; 
       unsigned int          m_device ; 
       unsigned int          m_buffer_id ;
       const char*           m_buffer_name ;
       CResource::Access_t   m_access ; 
       unsigned int          m_width  ; 
       unsigned int          m_height ; 
       unsigned int          m_depth ; 
       unsigned int          m_size ; 
       RTformat              m_format ; 
       unsigned int          m_type ; 
       void*                 m_dptr ;
       BufSpec               m_bufspec ; 
       Mapping_t             m_mapping ; 

};


inline OBuffer::OBuffer(optix::Context& context, unsigned int buffer_id, const char* buffer_name, unsigned int count, RTformat format, unsigned int type, CResource::Access_t access ) :
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
   m_format(format),
   m_type(type),
   m_bufspec(NULL,0,0),
   m_mapping(OBuffer::UNMAPPED)
{
   init();
}



/*
inline void* OBuffer::getRawPointer()
{
   return m_bufspec.dev_ptr ;
}
inline unsigned int OBuffer::getSize()
{
   return m_size ; 
}
inline BufSpec OBuffer::getBufSpec()
{
   return m_bufspec ; 
}
*/


inline bool OBuffer::isMapped()
{
   return m_mapping != UNMAPPED  ; 
}


