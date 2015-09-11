#pragma once

#include "OBufBase.hh"

// anything not using or related to template type should go in OBufBase

class OBuf : public OBufBase {
   public:
      OBuf( const char* name, optix::Buffer& buffer);
   public:

      template <typename T>
      T* getDevicePtr();

      template <typename T>
      void dump(const char* msg, unsigned int stride, unsigned int begin, unsigned int end );

      template <typename T>
      T reduce(unsigned int stride, unsigned int begin, unsigned int end=0u );


};


inline OBuf::OBuf(const char* name, optix::Buffer& buffer) 
   :
   OBufBase(name, buffer)
{
}



