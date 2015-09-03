#pragma once

#include <iostream>
#include <string>

#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

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
       template <typename F> void thrust_transform_index(const F& f, bool mapunmap);
       template <typename F> void thrust_transform_value(const F& f, bool mapunmap);
   public:
       // pertain to buffer from last get*Ptr call, so get the Ptr 1st  
       unsigned int          getBufferSize();
       unsigned int          getCount();
   public:
       void                  mapResources();
       void                  unmapResources();
       // get*Ptr() must be bookended between mapResources/unmapResources 
       T*                    getRawPtr();   
       thrust::device_ptr<T> getThrustPtr();   
       //CUdeviceptr           getCuPtr();   

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
    m_mapped = true ; 
}
template <typename T>
inline void CudaGLBuffer<T>::unmapResources()
{
    assert(m_mapped == true);
    unsigned int count = 1 ; 
    cudaError_t error = cudaGraphicsUnmapResources(count, &m_graphics_resource, m_stream);
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

template <typename T>
inline thrust::device_ptr<T> CudaGLBuffer<T>::getThrustPtr() 
{
    T* d_ptr = getRawPtr();
    thrust::device_ptr<T> th_ptr = thrust::device_pointer_cast(d_ptr);
    return th_ptr ; 
}


/*
template <typename T>
inline CUdeviceptr CudaGLBuffer<T>::getCuPtr() 
{
    T* d_ptr = getRawPtr();
    CUdeviceptr cu_ptr = reinterpret_cast<CUdeviceptr>(d_ptr) ; 
    return cu_ptr ; 
    // https://devtalk.nvidia.com/default/topic/551556/failed-on-optix-buffer-setdevicepointer-/
}
*/


template <typename T>
template <typename F>
inline void CudaGLBuffer<T>::thrust_transform_index(const F& f, bool mapunmap)
{
    if(mapunmap) mapResources();

    thrust::device_ptr<T> th_ptr = getThrustPtr();
    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last(m_count);
    thrust::transform(first, last, th_ptr,  f );

    if(mapunmap) unmapResources();
}


template <typename T>
template <typename F>
inline void CudaGLBuffer<T>::thrust_transform_value(const F& f, bool mapunmap)
{
    if(mapunmap) mapResources();

    thrust::device_ptr<T> ptr = getThrustPtr();
    thrust::device_vector<T> vec(ptr, ptr+m_count);

    thrust::transform(vec.begin(), vec.end(), vec.begin(),  f );
    printf("thrust_transform_value %d \n", m_count);

    if(mapunmap) unmapResources();
}












