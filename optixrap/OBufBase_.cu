#include "NPYBase.hpp"
#include "OBufBase.hh"

OBufBase::OBufBase(const char* name, optix::Buffer& buffer) 
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


CBufSlice OBufBase::slice( unsigned int stride, unsigned int begin, unsigned int end )
{
   return CBufSlice( getDevicePtr(), getSize(), getNumBytes(), stride, begin, end == 0u ? getNumAtoms() : end);
}

CBufSpec OBufBase::bufspec()
{
   return CBufSpec( getDevicePtr(), getSize(), getNumBytes()) ;
}


void OBufBase::Summary(const char* msg)
{
    printf("%s name %s size %u multiplicity %u sizeofatom %u NumAtoms %u NumBytes %u \n", msg, m_name, m_size, m_multiplicity, m_sizeofatom, getNumAtoms(), m_numbytes );
}

void OBufBase::setHexDump(bool hexdump)
{
   m_hexdump = hexdump ; 
}

/*
   *getSize()*  Excludes multiplicity of the type of the OptiX buffer, ie the size
                is the number of float4 

         Examples:

          1) Cerenkov genstep NPY<float> buffer with dimensions (7836,6,4)
             is canonically represented as an OptiX float4 buffer of size 7836*6 = 47016 

          2) Torch genstep NPY<float> buffer with dimensions (1,6,4)
             is canonically represented as an OptiX float4 buffer of size 1*6 = 6 

*/

unsigned int OBufBase::getSize()  
{
    return m_size ; 
}
unsigned int OBufBase::getMultiplicity()
{
    return m_multiplicity ; 
}
unsigned int OBufBase::getNumAtoms()
{
    return m_size*m_multiplicity ; 
}
unsigned int OBufBase::getSizeOfAtom()
{
    return m_sizeofatom ; 
}
unsigned int OBufBase::getNumBytes()
{
    return m_numbytes ; 
}






void OBufBase::init()
{
    examineBufferFormat(m_buffer->getFormat());
    m_size = getSize(m_buffer);
    m_numbytes = getNumBytes(m_buffer);
}

void OBufBase::examineBufferFormat(RTformat format)
{
   unsigned int mul(0) ;
   unsigned int soa(0) ;
   switch(format)
   {   
      case RT_FORMAT_UNKNOWN: mul=0 ;soa=0 ;  break ; 

      case RT_FORMAT_FLOAT:   mul=1 ; soa=sizeof(float) ; break ;
      case RT_FORMAT_FLOAT2:  mul=2 ; soa=sizeof(float) ; break ;
      case RT_FORMAT_FLOAT3:  mul=3 ; soa=sizeof(float) ; break ;
      case RT_FORMAT_FLOAT4:  mul=4 ; soa=sizeof(float) ; break ;

      case RT_FORMAT_BYTE:    mul=1 ; soa=sizeof(char)  ; break ;
      case RT_FORMAT_BYTE2:   mul=2 ; soa=sizeof(char)  ; break ;
      case RT_FORMAT_BYTE3:   mul=3 ; soa=sizeof(char)  ; break ;
      case RT_FORMAT_BYTE4:   mul=4 ; soa=sizeof(char)  ; break ;

      case RT_FORMAT_UNSIGNED_BYTE:  mul=1 ; soa=sizeof(unsigned char) ; break ;
      case RT_FORMAT_UNSIGNED_BYTE2: mul=2 ; soa=sizeof(unsigned char) ; break ;
      case RT_FORMAT_UNSIGNED_BYTE3: mul=3 ; soa=sizeof(unsigned char) ; break ;
      case RT_FORMAT_UNSIGNED_BYTE4: mul=4 ; soa=sizeof(unsigned char) ; break ;

      case RT_FORMAT_SHORT:  mul=1 ; soa=sizeof(short) ; break ;
      case RT_FORMAT_SHORT2: mul=2 ; soa=sizeof(short) ; break ;
      case RT_FORMAT_SHORT3: mul=3 ; soa=sizeof(short) ; break ;
      case RT_FORMAT_SHORT4: mul=4 ; soa=sizeof(short) ; break ;

      case RT_FORMAT_UNSIGNED_SHORT:  mul=1 ; soa=sizeof(unsigned short) ; break ;
      case RT_FORMAT_UNSIGNED_SHORT2: mul=2 ; soa=sizeof(unsigned short) ; break ;
      case RT_FORMAT_UNSIGNED_SHORT3: mul=3 ; soa=sizeof(unsigned short) ; break ;
      case RT_FORMAT_UNSIGNED_SHORT4: mul=4 ; soa=sizeof(unsigned short) ; break ;

      case RT_FORMAT_INT:  mul=1 ; soa=sizeof(int) ; break ;
      case RT_FORMAT_INT2: mul=2 ; soa=sizeof(int) ; break ;
      case RT_FORMAT_INT3: mul=3 ; soa=sizeof(int) ; break ;
      case RT_FORMAT_INT4: mul=4 ; soa=sizeof(int) ; break ;

      case RT_FORMAT_UNSIGNED_INT:  mul=1 ; soa=sizeof(unsigned int) ; break ;
      case RT_FORMAT_UNSIGNED_INT2: mul=2 ; soa=sizeof(unsigned int) ; break ;
      case RT_FORMAT_UNSIGNED_INT3: mul=3 ; soa=sizeof(unsigned int) ; break ;
      case RT_FORMAT_UNSIGNED_INT4: mul=4 ; soa=sizeof(unsigned int) ; break ;

      case RT_FORMAT_USER:       mul=0 ; soa=0 ; break ;
      case RT_FORMAT_BUFFER_ID:  mul=0 ; soa=0 ; break ;
      case RT_FORMAT_PROGRAM_ID: mul=0 ; soa=0 ; break ; 
   }   

    unsigned int element_size_bytes = getElementSizeInBytes(format);
    assert(element_size_bytes == soa*mul );

    setMultiplicity(mul)  ;
    setSizeOfAtom(soa) ;
}


void OBufBase::setSizeOfAtom(unsigned int soa)
{
    m_sizeofatom = soa ; 
} 
void OBufBase::setMultiplicity(unsigned int mul)
{
    m_multiplicity = mul ; 
} 


unsigned int OBufBase::getElementSizeInBytes(RTformat format)
{
    size_t element_size ; 
    rtuGetSizeForRTformat( format, &element_size);
    return element_size ; 
}

void* OBufBase::getDevicePtr()
{
    CUdeviceptr cu_ptr = m_buffer->getDevicePointer(m_device) ;
    return (void*)cu_ptr ; 
}

unsigned int OBufBase::getSize(const optix::Buffer& buffer)
{
    RTsize width, height, depth ; 
    buffer->getSize(width, height, depth);
    RTsize size = width*height*depth ; 
    return size ; 
}

unsigned int OBufBase::getNumBytes(const optix::Buffer& buffer)
{
    unsigned int size = getSize(buffer);

    RTformat format = buffer->getFormat() ;
    unsigned int element_size = getElementSizeInBytes(format);
    if(element_size == 0u && format == RT_FORMAT_USER)
    {
        element_size = buffer->getElementSize();
        printf("OBufBase::getNumBytes RT_FORMAT_USER element_size %u size %u \n", element_size, size );
    }
    return size*element_size ; 
}

void OBufBase::upload(NPYBase* npy)
{
    void* data = npy->getBytes() ;

    assert(data);

    unsigned int numBytes = npy->getNumBytes(0);

    printf("OBufBase::upload nbytes %u \n", numBytes);

    memcpy( m_buffer->map(), data, numBytes );

    m_buffer->unmap();
}


void OBufBase::download(NPYBase* npy)
{
    unsigned int numBytes = npy->getNumBytes(0) ;

    assert(numBytes == m_numbytes);

    void* ptr = m_buffer->map() ; 

    npy->read( ptr );

    m_buffer->unmap(); 
}

