#include "OBufBase.hh"


void OBufBase::init()
{
    m_size = getSize(m_buffer);
    m_element_size = getElementSizeInBytes(m_buffer)/m_atom_size ;
}


unsigned int OBufBase::getElementSizeInBytes(const optix::Buffer& buffer)
{
    size_t element_size ; 
    rtuGetSizeForRTformat( buffer->getFormat(), &element_size);
    return element_size ; 
}


unsigned int OBufBase::getSize(const optix::Buffer& buffer)
{
    RTsize width, height, depth ; 
    buffer->getSize(width, height, depth);
    RTsize size = width*height*depth ; 
    return size ; 
}


