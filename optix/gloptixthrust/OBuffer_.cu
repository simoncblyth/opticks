
#include <optix_world.h>

#include <iostream>
#include <iterator>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "OBuffer.hh"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "helper_cuda.h"


struct Resource {
   unsigned int buffer_id ; 
   size_t       bufsize  ; 
   unsigned int flags ; 
   cudaStream_t stream ; 
   struct cudaGraphicsResource*  resource ;
   void*         dev_ptr ;   

   Resource(unsigned int buffer_id, unsigned int flags, cudaStream_t stream) : 
       buffer_id(buffer_id),
       bufsize(0),
       flags(flags),  
       stream(stream),
       resource(NULL),
       dev_ptr(NULL)
   {
   }

   const char* getFlagDescription()
   {
       const char* ret(NULL);
       switch(flags)
       {
           case cudaGraphicsMapFlagsNone:         ret="cudaGraphicsMapFlagsNone: Default; Assume resource can be read/written " ; break ;
           case cudaGraphicsMapFlagsReadOnly:     ret="cudaGraphicsMapFlagsReadOnly: CUDA will not write to this resource " ; break ; 
           case cudaGraphicsMapFlagsWriteDiscard: ret="cudaGraphicsMapFlagsWriteDiscard: CUDA will only write to and will not read from this resource " ; break ;  
       }
       return ret ;
   }

   void registerBuffer()
   {
       printf("Resource::registerBuffer %d : %s \n", buffer_id, getFlagDescription() );
       checkCudaErrors( cudaGraphicsGLRegisterBuffer(&resource, buffer_id, flags) );
   }

   void unregisterBuffer()
   {
       printf("Resource::unregisterBuffer %d \n", buffer_id );
       checkCudaErrors( cudaGraphicsUnregisterResource(resource) );
   }


   void* mapGLToCUDA() 
   {
       checkCudaErrors( cudaGraphicsMapResources(1, &resource, stream) );
       checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void **)&dev_ptr, &bufsize, resource) );
       printf("Resource::mapGLToCUDA bufsize %lu dev_ptr %p \n", bufsize, dev_ptr );
       return dev_ptr ; 
   }

   void unmapGLToCUDA()
   {
       printf("Resource::unmapGLToCUDA\n");
       checkCudaErrors( cudaGraphicsUnmapResources(1, &resource, stream));
   } 

   void streamSync()
   {
       printf("Resource::streamSync\n");
       checkCudaErrors( cudaStreamSynchronize(stream));
   }

};


void OBuffer::init()
{
    unsigned int flags ;
    switch(m_access)
    {
       case RW: flags = cudaGraphicsMapFlagsNone         ;break;
       case  R: flags = cudaGraphicsMapFlagsReadOnly     ;break;
       case  W: flags = cudaGraphicsMapFlagsWriteDiscard ;break;
    }

    cudaStream_t stream1 ; 
    cudaStreamCreate ( &stream1) ;
    m_resource = new Resource(m_buffer_id, flags, stream1  );
}

void OBuffer::streamSync()
{
    m_resource->streamSync();
}

void OBuffer::mapGLToCUDA()
{
    m_mapped = true ; 
    printf("OBuffer::mapGLToCUDA %d\n", m_buffer_id);
    m_resource->registerBuffer();
    m_dptr = m_resource->mapGLToCUDA();
}

void OBuffer::unmapGLToCUDA()
{
    m_mapped = false ; 
    printf("OBuffer::unmapGLToCUDA\n");
    m_resource->unmapGLToCUDA();
    m_resource->unregisterBuffer();
}

void OBuffer::mapGLToCUDAToOptiX()
{
    printf("OBuffer::mapGLToCUDAToOptiX  getMappedPointer,createBufferForCUDA,setDevicePointer \n");
    mapGLToCUDA();
    CUdeviceptr cu_ptr = reinterpret_cast<CUdeviceptr>(m_dptr) ; 
    m_buffer = m_context->createBufferForCUDA(m_type, m_format, m_size);
    m_buffer->setDevicePointer(m_device, cu_ptr );
    m_context[m_buffer_name]->setBuffer(m_buffer);
}
void OBuffer::unmapGLToCUDAToOptiX()
{
    printf("OBuffer::unmapGLToCUDAToOptiX\n");
    m_buffer->markDirty();  // before the unmap : think of the unmap as potentially copying back to GL
    unmapGLToCUDA();
}

void OBuffer::mapGLToOptiX()
{
    printf("OBuffer::mapGLToOptiX (createBufferFromGLBO) %d\n", m_buffer_id);
    m_buffer = m_context->createBufferFromGLBO(m_type, m_buffer_id);

    //m_buffer->registerGLBuffer();

    m_buffer->setFormat( m_format );
    m_buffer->setSize( m_size );
    m_context[m_buffer_name]->setBuffer(m_buffer);
}

void OBuffer::mapOptiXToCUDA()
{
   /*

When the application requests a pointer from OptiX (to an RT_BUFFER_INPUT or
RT_BUFFER_INPUT_OUTPUT buffer), we assume that the application is modifying the
data contained in that buffer. Therefore we keep track of which OptiX devices
the application has requested pointers for, and if the application has
requested only one pointer but there are additional OptiX devices, we will copy
the data from that device to all others on the next launch. If the application
requests pointers on all devices, we assume they have set up the data how they
want it, and no copying will happen. It is a caught runtime error to request
pointers for more than one but fewer than all devices.

   */


    CUdeviceptr d;
    m_buffer->getDevicePointer(m_device, &d);
    m_dptr = (void*)(d) ;
    printf("OBuffer::mapOptiXToCUDA (getDevicePointer) dptr %p \n", m_dptr);
}
void OBuffer::unmapOptiXToCUDA()
{
    printf("OBuffer::unmapOptiXToCUDA (markDirty) \n");
    m_buffer->markDirty(); 
}


void OBuffer::unmapGLToOptiX()
{
    //m_buffer->unregisterGLBuffer();
    printf("OBuffer::unmapGLToOptiX (noop)\n");
}


