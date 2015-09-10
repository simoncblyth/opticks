#pragma once

#include <optix_world.h>
#include <optixu/optixpp_namespace.h>

template <typename T>
class OBuf {
   public:
      OBuf( optix::Buffer& buffer );
      T* getDevicePtr();
      unsigned int getSize();
      void dump(const char* msg, unsigned int begin, unsigned int end );
   private:
      optix::Buffer  m_buffer  ;
      unsigned int   m_device ; 
};



