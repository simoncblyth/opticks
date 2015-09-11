#pragma once

#include "OBufBase.hh"
#include "OBufSlice.hh"



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

      OBufSlice slice( unsigned int stride, unsigned int begin, unsigned int end=0u );
};

inline OBuf::OBuf(const char* name, optix::Buffer& buffer) 
   :
   OBufBase(name, buffer)
{
}

inline OBufSlice OBuf::slice( unsigned int stride, unsigned int begin, unsigned int end )
{
   return OBufSlice(this, stride, begin, end == 0u ? getNumAtoms() : end);
}



