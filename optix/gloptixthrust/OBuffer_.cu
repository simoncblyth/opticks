
#include <optix_world.h>

#include <iostream>
#include <iterator>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "OBuffer.hh"
#include "CResource.hh"

#include <cuda_runtime.h>


void OBuffer::init()
{
    if(m_inited) return ;
    m_inited = true ; 

    printf("OBuffer::init %u %s \n", m_buffer_id, m_buffer_name );
    if(m_buffer_id == 0) return ; 

    m_resource = new CResource(m_buffer_id, m_access  );
}

unsigned int OBuffer::getNumBytes()
{
    return m_resource ? m_resource->getNumBytes() : 0 ; 
}

void OBuffer::streamSync()
{
    assert(m_resource);
    m_resource->streamSync();
}

void OBuffer::mapGLToCUDA()
{
    assert(m_resource);
    m_mapped = true ; 
    printf("OBuffer::mapGLToCUDA %d\n", m_buffer_id);
    m_resource->mapGLToCUDA();

    m_bufspec.dev_ptr = m_resource->getRawPointer() ;
    m_bufspec.size    = m_size ; 
    m_bufspec.num_bytes = m_resource->getNumBytes() ; 
}

void OBuffer::unmapGLToCUDA()
{
    assert(m_resource);
    m_mapped = false ; 
    printf("OBuffer::unmapGLToCUDA\n");
    m_resource->unmapGLToCUDA();
}

void OBuffer::mapGLToCUDAToOptiX()
{
    printf("OBuffer::mapGLToCUDAToOptiX  getMappedPointer,createBufferForCUDA,setDevicePointer \n");
    m_resource->mapGLToCUDA();
    void* dptr = m_resource->getRawPointer();

    CUdeviceptr cu_ptr = reinterpret_cast<CUdeviceptr>(dptr) ; 
    m_buffer = m_context->createBufferForCUDA(m_type, m_format, m_size);
    m_buffer->setDevicePointer(m_device, cu_ptr );

    fillBufSpec( dptr );
    assert( m_bufspec.num_bytes == m_resource->getNumBytes()); 

    m_context[m_buffer_name]->setBuffer(m_buffer);
}

void OBuffer::unmapGLToCUDAToOptiX()
{
    printf("OBuffer::unmapGLToCUDAToOptiX\n");
    m_buffer->markDirty();  // before the unmap : think of the unmap as potentially copying back to GL
    unmapGLToCUDA();
}

void OBuffer::create()
{
    assert(m_buffer_id == 0 );
    printf("OBuffer::create %s (createBuffer) %d  size %d\n", m_buffer_name, m_buffer_id, m_size);
    m_buffer = m_context->createBuffer(m_type, m_format, m_size);
    fillBufSpec( NULL );
    m_context[m_buffer_name]->setBuffer(m_buffer);
}

void OBuffer::mapGLToOptiX()
{
    assert(m_resource);
    printf("OBuffer::mapGLToOptiX %s (createBufferFromGLBO) %d  size %d\n", m_buffer_name, m_buffer_id, m_size);
    m_buffer = m_context->createBufferFromGLBO(m_type, m_buffer_id);
    m_buffer->setFormat( m_format );
    m_buffer->setSize( m_size );

    fillBufSpec( NULL );

    m_context[m_buffer_name]->setBuffer(m_buffer);
}
void OBuffer::unmapGLToOptiX()
{
    printf("OBuffer::unmapGLToOptiX (noop)\n");
}

void OBuffer::mapOptiXToCUDA()
{
    CUdeviceptr dev_ptr;
    m_buffer->getDevicePointer(m_device, &dev_ptr);

    fillBufSpec( (void*)dev_ptr );

    m_bufspec.Summary("OBuffer::mapOptiXToCUDA (getDevicePointer) bufspec");
}

void OBuffer::unmapOptiXToCUDA()
{
    printf("OBuffer::unmapOptiXToCUDA (markDirty) \n");  // when is this acted upon ? next launch perhaps ? need dummy launch maybe
    m_buffer->markDirty(); 
}


unsigned int OBuffer::getBufferSize()
{
    RTsize width, height, depth ; 
    m_buffer->getSize(width, height, depth);
    unsigned int size = width*height*depth ; 
    assert(size == m_size );
    return size ; 
}
unsigned int OBuffer::getElementSize()
{
    size_t element_size ; 
    rtuGetSizeForRTformat( m_buffer->getFormat(), &element_size);
    return element_size ; 
}


void OBuffer::fillBufSpec(void* dev_ptr)
{
    unsigned int element_size = getElementSize();
    unsigned int size = getBufferSize();

    m_bufspec.dev_ptr = dev_ptr ; 
    m_bufspec.size = size ; 
    m_bufspec.num_bytes = element_size*size  ;
}


