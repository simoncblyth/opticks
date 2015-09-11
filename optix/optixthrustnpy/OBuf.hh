#pragma once

#include "OBufBase.hh"


// anything not using or related to template type should go in OBufBase

template <typename T>
class OBuf : public OBufBase {
   public:
      OBuf( const char* name, optix::Buffer& buffer );
   public:
      T* getDevicePtr();
      void dump(const char* msg, unsigned int begin, unsigned int end );
      void dump_strided(const char* msg, unsigned int begin, unsigned int end, unsigned int stride);
};



template <typename T>
inline OBuf<T>::OBuf(const char* name, optix::Buffer& buffer ) 
   :
   OBufBase(name, buffer)
{
}



