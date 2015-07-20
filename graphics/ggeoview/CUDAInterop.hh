#pragma once

//  https://gist.github.com/dangets/2926425
//  /usr/local/env/cuda/NVIDIA_CUDA-7.0_Samples/2_Graphics/simpleGL/simpleGL.cu
//  file:///Developer/NVIDIA/CUDA-7.0/doc/html/cuda-c-programming-guide/index.html#graphics-interoperability

#include "NPY.hpp"

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


template<typename T>
class CUDAInterop {
   public:
       static void init();
       CUDAInterop(NPY<T>* npy);
   public:
       int  getBufferId();
       void registerBuffer();
       T*   GL_to_CUDA();
       void CUDA_to_GL();
   private:
       NPY<T>*                      m_npy ; 
       struct cudaGraphicsResource* m_vbo_cuda; 
       T*                           m_raw_ptr ;        
       size_t                       m_buf_size ;  
       bool                         m_registered ; 

};




template <typename T>
inline void CUDAInterop<T>::init()
{
    cudaGLSetGLDevice(0);
}


template <typename T>
inline CUDAInterop<T>::CUDAInterop(NPY<T>* npy)
       :
       m_npy(npy),
       m_vbo_cuda(NULL),
       m_raw_ptr(NULL),
       m_buf_size(0),
       m_registered(false)
{
   npy->setAux(this);
} 


template<typename T> 
inline int CUDAInterop<T>::getBufferId()
{
    return m_npy ? m_npy->getBufferId() : -1 ; 
}

template<typename T> 
inline void CUDAInterop<T>::registerBuffer()
{
    if(m_registered)
    {
        LOG(info) << "CUDAInterop<T>::registerBuffer already done " ;
        return ; 
    }
    int buffer_id = getBufferId();

    if(buffer_id < 0)
    {
        LOG(info) << "CUDAInterop<T>::CUDAInterop cannot register buffer now as not yet introduced to OpenGL " ;
        return ;
    }

    LOG(info) << "CUDAInterop<T>::registerBuffer "  << buffer_id ;


    //unsigned int flags = cudaGraphicsRegisterFlagsNone ;     // no hints, it is assumed to be read from and written to by CUDA
    //unsigned int flags = cudaGraphicsRegisterFlagsReadOnly ; // CUDA will not write to this resource
    unsigned int flags = cudaGraphicsRegisterFlagsWriteDiscard ; // CUDA will not read (discard prior contents) and will write all over it 

    cudaGraphicsGLRegisterBuffer(&m_vbo_cuda, buffer_id, flags );
    m_registered = true ; 
}

template <typename T>
inline T* CUDAInterop<T>::GL_to_CUDA()
{
    assert(m_registered);
    LOG(info) << "CUDAInterop<T>::GL_to_CUDA " << getBufferId() ; 
    cudaGraphicsMapResources(1, &m_vbo_cuda, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&m_raw_ptr, &m_buf_size, m_vbo_cuda);
    return m_raw_ptr ; 
}

template <typename T>
inline void CUDAInterop<T>::CUDA_to_GL()
{
    LOG(info) << "CUDAInterop<T>::CUDA_to_CL " << getBufferId() ; 
    cudaGraphicsUnmapResources(1, &m_vbo_cuda, 0);
}


