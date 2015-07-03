#pragma once

//  https://gist.github.com/dangets/2926425

#include "NPY.hpp"

#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


template<typename T>
class CUDAInterop {
   public:
       static void enable(NPYBase* npy);
   public:
       CUDAInterop();
   public:
       void registerBuffer( unsigned int buffer_id );
       T*   GL_to_CUDA();
       void CUDA_to_GL();
   private:
       T*   getRawPointer();
     
   private:
       struct cudaGraphicsResource* m_vbo_cuda; 
       T*                           m_raw_ptr ;        
       size_t                       m_buf_size ;  

};


template<typename T> 
inline void CUDAInterop<T>::enable(NPYBase* npy)
{
    LOG(info)<<"CUDAInterop<T>::enable " << npy->description() ; 
    npy->setAux(new CUDAInterop<T>);
}


template <typename T>
inline CUDAInterop<T>::CUDAInterop()
       :
       m_vbo_cuda(NULL),
       m_raw_ptr(NULL),
       m_buf_size(0)
       {
       } 


template<typename T> 
inline void CUDAInterop<T>::registerBuffer( unsigned int buffer_id )
{
    cudaGraphicsGLRegisterBuffer(&m_vbo_cuda, buffer_id, cudaGraphicsMapFlagsWriteDiscard);
}

template <typename T>
inline T* CUDAInterop<T>::GL_to_CUDA()
{
    cudaGraphicsMapResources(1, &m_vbo_cuda, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&m_raw_ptr, &m_buf_size, m_vbo_cuda);
    return m_raw_ptr ; 
}

template <typename T>
inline T* CUDAInterop<T>::getRawPointer()
{
     return m_raw_ptr ;
}

template <typename T>
inline void CUDAInterop<T>::CUDA_to_GL()
{
     cudaGraphicsUnmapResources(1, &m_vbo_cuda, 0);
}


