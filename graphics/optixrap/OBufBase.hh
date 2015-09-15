#pragma once
#include <optix_world.h>
#include <optixu/optixpp_namespace.h>

#include "CBufSlice.hh"
class NPYBase ; 


// non-view-type specifics
class OBufBase {
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
      CBufSlice slice( unsigned int stride, unsigned int begin, unsigned int end=0u );
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


inline CBufSlice OBufBase::slice( unsigned int stride, unsigned int begin, unsigned int end )
{
   return CBufSlice( getDevicePtr(), getSize(), getNumBytes(), stride, begin, end == 0u ? getNumAtoms() : end);
}

inline void OBufBase::Summary(const char* msg)
{
    printf("%s name %s size %u multiplicity %u sizeofatom %u NumAtoms %u NumBytes %u \n", msg, m_name, m_size, m_multiplicity, m_sizeofatom, getNumAtoms(), m_numbytes );
}



inline OBufBase::OBufBase(const char* name, optix::Buffer& buffer) 
   :
   m_buffer(buffer), 
   m_name(strdup(name)), 
   m_size(0u),
   m_multiplicity(0u), 
   m_sizeofatom(0u), 
   m_numbytes(0u), 
   m_device(0u),
   m_hexdump(false)
{
    init();
}


inline void OBufBase::setHexDump(bool hexdump)
{
   m_hexdump = hexdump ; 
}


/*
   *getSize()* Excludes multiplicity of the type of the OptiX buffer

        e.g a Cerenkov genstep NPY<float> buffer with dimensions (7836,6,4)
        is canonically represented as an OptiX float4 buffer of size 7836*6 = 47016 
*/

inline unsigned int OBufBase::getSize()  
{
    return m_size ; 
}
inline unsigned int OBufBase::getMultiplicity()
{
    return m_multiplicity ; 
}
inline unsigned int OBufBase::getNumAtoms()
{
    return m_size*m_multiplicity ; 
}
inline unsigned int OBufBase::getSizeOfAtom()
{
    return m_sizeofatom ; 
}
inline unsigned int OBufBase::getNumBytes()
{
    return m_numbytes ; 
}





