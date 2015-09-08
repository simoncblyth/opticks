
#include "CResource.hh"
#include "assert.h"
#include <cuda_gl_interop.h>
#include "helper_cuda.h"

struct CResourceImp {
   unsigned int buffer_id ; 
   size_t       bufsize  ; 
   unsigned int flags ; 
   cudaStream_t stream ; 
   struct cudaGraphicsResource*  resource ;
   void*         dev_ptr ;   

   CResourceImp(unsigned int buffer_id, unsigned int flags, cudaStream_t stream) : 
       buffer_id(buffer_id),
       bufsize(0),
       flags(flags),  
       stream(NULL),
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
       //printf("Resource::registerBuffer %d : %s \n", buffer_id, getFlagDescription() );
       checkCudaErrors( cudaGraphicsGLRegisterBuffer(&resource, buffer_id, flags) );
   }

   void unregisterBuffer()
   {
       //printf("Resource::unregisterBuffer %d \n", buffer_id );
       checkCudaErrors( cudaGraphicsUnregisterResource(resource) );
   }


   void* mapGLToCUDA() 
   {
       checkCudaErrors( cudaGraphicsMapResources(1, &resource, stream) );
       checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void **)&dev_ptr, &bufsize, resource) );
       //printf("Resource::mapGLToCUDA bufsize %lu dev_ptr %p \n", bufsize, dev_ptr );
       return dev_ptr ; 
   }

   void unmapGLToCUDA()
   {
       //printf("Resource::unmapGLToCUDA\n");
       checkCudaErrors( cudaGraphicsUnmapResources(1, &resource, stream));
   } 

   void streamSync()
   {
       //printf("Resource::streamSync\n");
       checkCudaErrors( cudaStreamSynchronize(stream));
   }

};


void CResource::init()
{
    unsigned int flags ;
    switch(m_access)
    {
       case RW: flags = cudaGraphicsMapFlagsNone         ;break;
       case  R: flags = cudaGraphicsMapFlagsReadOnly     ;break;
       case  W: flags = cudaGraphicsMapFlagsWriteDiscard ;break;
    }

    //cudaStream_t stream1 ; 
    //cudaStreamCreate ( &stream1) ;
    m_imp = new CResourceImp(m_buffer_id, flags, (cudaStream_t)0  );
}


unsigned int CResource::getNumBytes()
{
    assert(m_mapped);
    return m_imp->bufsize ; 
}

void* CResource::getRawPointer()
{
    assert(m_mapped);
    return m_imp->dev_ptr ;
}

void CResource::streamSync()
{
    m_imp->streamSync();
}
void CResource::mapGLToCUDA()
{
    m_mapped = true ; 
    //printf("CResource::mapGLToCUDA %d\n", m_buffer_id);
    m_imp->registerBuffer();
    m_imp->mapGLToCUDA();
}
void CResource::unmapGLToCUDA()
{
    m_mapped = false ; 
    //printf("CResource::unmapGLToCUDA\n");
    m_imp->unmapGLToCUDA();
    m_imp->unregisterBuffer();
}


