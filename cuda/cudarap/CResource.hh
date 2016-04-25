#pragma once
#include "stdlib.h"
#include "CBufSpec.hh"

struct CResourceImp ; 

// OpenGL buffer made available as a CUDA Resource 
class CResource {
    public:
       typedef enum { RW, R, W } Access_t ; 
    public:
       CResource( unsigned int buffer_id, Access_t access );
    private:
       void init(); 
    public:
       template <typename T> CBufSpec mapGLToCUDA();
       void unmapGLToCUDA();
    public:
       void streamSync();
    private:
       CResourceImp*  m_imp ; 
       unsigned int   m_buffer_id ; 
       Access_t       m_access ; 
       bool           m_mapped ; 
};


inline CResource::CResource(unsigned int buffer_id, Access_t access ) :
   m_imp(NULL),
   m_buffer_id(buffer_id),
   m_access(access),
   m_mapped(false)
{
   init();
}



