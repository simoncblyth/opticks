#pragma once

#include <iostream>
#include <string>
#include <cuda_gl_interop.h>

// this a a cut down version compared to original in glfwtriangle- 
template <class T>
class CudaGLBuffer {
   public:
       CudaGLBuffer(
           unsigned int buffer_id, 
           unsigned int flags, 
           cudaStream_t stream=0);
   public:
       const char* getFlagDescription();
       void Summary(const char* msg="InteropBuffer::Summary");
   public:
       void                  mapResources();
       T*                    getRawPtr();      // must be bookended between mapResources/unmapResources 
       void                  unmapResources();
   public:
       // pertain to buffer from last getRawPtr call, so do that first
       unsigned int          getBufferSize();
       unsigned int          getCount();
   private:  
       unsigned int                 m_buffer_id ;   
   private:  
       unsigned int                 m_flags ;   
       cudaStream_t                 m_stream ; 
       bool                         m_mapped ; 
       struct cudaGraphicsResource* m_graphics_resource ;
   private:
       size_t                       m_bufsize ; // dimensions obtained by getRawPtr
       unsigned int                 m_count ; 
};


template <typename T>
inline CudaGLBuffer<T>::CudaGLBuffer(
           unsigned int buffer_id, 
           unsigned int flags, 
           cudaStream_t stream)
     :
     m_buffer_id(buffer_id),
     m_flags(flags),
     m_stream(stream),
     m_mapped(false),
     m_bufsize(0),
     m_count(0)
{
    cudaGraphicsGLRegisterBuffer(&m_graphics_resource, m_buffer_id, m_flags);
}




template <typename T>
inline const char* CudaGLBuffer<T>::getFlagDescription()
{
    const char* ret(NULL);
    switch(m_flags)
    {
        case cudaGraphicsMapFlagsNone:         ret="cudaGraphicsMapFlagsNone: Default; Assume resource can be read/written " ; break ;
        case cudaGraphicsMapFlagsReadOnly:     ret="cudaGraphicsMapFlagsReadOnly: CUDA will not write to this resource " ; break ; 
        case cudaGraphicsMapFlagsWriteDiscard: ret="cudaGraphicsMapFlagsWriteDiscard: CUDA will only write to and will not read from this resource " ; break ;  
    }
    return ret ;
}



template <typename T>
inline unsigned int CudaGLBuffer<T>::getCount()
{
    return m_count ; 
}

template <typename T>
inline unsigned int CudaGLBuffer<T>::getBufferSize()
{
    return m_bufsize ; 
}


template <typename T>
inline void CudaGLBuffer<T>::mapResources()
{
    assert(m_mapped == false);
    unsigned int count = 1 ; 
    cudaError_t error = cudaGraphicsMapResources(count, &m_graphics_resource, m_stream);
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    m_mapped = true ; 
}
template <typename T>
inline void CudaGLBuffer<T>::unmapResources()
{
    assert(m_mapped == true);
    unsigned int count = 1 ; 
    cudaError_t error = cudaGraphicsUnmapResources(count, &m_graphics_resource, m_stream);
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    m_mapped = false ; 
}
template <typename T>
inline void CudaGLBuffer<T>::Summary(const char* msg)
{
    printf("%s buffer_id %d bufsize %lu count %d \n%s\n", msg, m_buffer_id, m_bufsize, m_count, getFlagDescription() );
}

template <typename T>
inline T* CudaGLBuffer<T>::getRawPtr() 
{
    assert(m_mapped == true);

    T* d_ptr;
    cudaGraphicsResourceGetMappedPointer((void **)&d_ptr, &m_bufsize, m_graphics_resource);

    m_count = m_bufsize/sizeof(T) ;  // 1d assumption

    return d_ptr ; 
}







