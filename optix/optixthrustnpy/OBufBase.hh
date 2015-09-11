#pragma once
#include <optix_world.h>
#include <optixu/optixpp_namespace.h>

// non-view-type specifics
class OBufBase {
   public:
      OBufBase( const char* name, optix::Buffer& buffer, unsigned int atom_size=4 );
   private:
      void init();
   public:
      unsigned int getElementSize(); // typically 4, for RT_FORMAT_FLOAT4/RT_FORMAT_UINT4
      unsigned int getSize();        // number of elements (usually float4) in OptiX buffer 
      unsigned int getAtomicSize();  // ElementSize * Size, giving number of atoms, usually floats
   public:
      void Summary(const char* msg="OBufBase::Summary");
   public:
      static unsigned int getSize(const optix::Buffer& buffer);
      static unsigned int getElementSizeInBytes(const optix::Buffer& buffer); // eg sizeof(RT_FORMAT_FLOAT4) = 4*4 = 16 
   protected:
      optix::Buffer  m_buffer  ;
      const char*    m_name ; 
      unsigned int   m_size ; 
      unsigned int   m_element_size ; 
      unsigned int   m_atom_size ; 
      unsigned int   m_device ; 
};

inline void OBufBase::Summary(const char* msg)
{
    printf("%s name %s size %u element_size %u atomic_size %u \n", msg, m_name, m_size, m_element_size, getAtomicSize() );
}



inline OBufBase::OBufBase(const char* name, optix::Buffer& buffer, unsigned int atom_size ) 
   :
   m_buffer(buffer), 
   m_name(strdup(name)), 
   m_size(0u),
   m_element_size(0u),
   m_atom_size(atom_size), 
   m_device(0u) 
{
    init();
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
inline unsigned int OBufBase::getElementSize()
{
    return m_element_size ; 
}
inline unsigned int OBufBase::getAtomicSize()
{
    return m_size*m_element_size ; 
}




