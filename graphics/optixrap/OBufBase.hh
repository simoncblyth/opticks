#pragma once
// NB implementation in OBufBase_.cu as requires nvcc compilation
#include <optix_world.h>
#include <optixu/optixpp_namespace.h>

// cudawrap- struct 
#include "CBufSlice.hh"
#include "CBufSpec.hh"
class NPYBase ; 

// non-view-type specifics
#include "OXRAP_API_EXPORT.hh"

class OXRAP_API OBufBase {
   public:
      OBufBase( const char* name, optix::Buffer& buffer );
   public:
      void upload(NPYBase* npy);
      void download(NPYBase* npy);
      void setHexDump(bool hexdump);
   private:
      void init();
      void examineBufferFormat(RTformat format);
      static unsigned int getElementSizeInBytes(RTformat format); // eg sizeof(RT_FORMAT_FLOAT4) = 4*4 = 16 
      static unsigned int getNumBytes(const optix::Buffer& buffer);
   public:
      CBufSpec  bufspec();
      CBufSlice slice( unsigned int stride, unsigned int begin=0u , unsigned int end=0u );
      void*        getDevicePtr();
      unsigned int getMultiplicity(); // typically 4, for RT_FORMAT_FLOAT4/RT_FORMAT_UINT4
      unsigned int getSize();         // width*depth*height of OptiX buffer, ie the number of typed elements (often float4) 
      unsigned int getNumAtoms();     // Multiplicity * Size, giving number of atoms, eg number of floats or ints
      unsigned int getSizeOfAtom();   // in bytes, eg 4 for any of the RT_FORMAT_FLOAT4 RT_FORMAT_FLOAT3 ... formats 
      unsigned int getNumBytes();     // total buffer size in bytes

   public:
      // usually set in ctor by examineBufferFormat, but RT_FORMAT_USER needs to be set manually 
      void setSizeOfAtom(unsigned int soa);
      void setMultiplicity(unsigned int mul);
   public:
      void Summary(const char* msg="OBufBase::Summary");
   public:
      static unsigned int getSize(const optix::Buffer& buffer);
   protected:
      optix::Buffer  m_buffer  ;
      const char*    m_name ; 
      unsigned int   m_size ; 
      unsigned int   m_multiplicity ; 
      unsigned int   m_sizeofatom ; 
      unsigned int   m_numbytes ; 
      unsigned int   m_device ; 
      bool           m_hexdump ; 
};


