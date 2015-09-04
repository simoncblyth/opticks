
//#include <cuda.h>
#include <optix_world.h>

#include <iostream>
#include <iterator>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "gloptixthrust.hh"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "helper_cuda.h"
#include "helper_cuda.h"


unsigned int g_flags = cudaGraphicsMapFlagsNone ;
//unsigned int g_flags = cudaGraphicsMapFlagsReadOnly ;
//unsigned int   g_flags = cudaGraphicsMapFlagsWriteDiscard ;
size_t         g_bufsize = 0 ; 
bool           g_registered = false ; 
unsigned int   g_buffer_id = 0 ; 
cudaStream_t   g_stream = 0 ; 
struct cudaGraphicsResource*  g_graphics_resource = NULL ;
void* g_dev_ptr = NULL ;


void setBufferId(unsigned int buffer_id)
{
    g_buffer_id = buffer_id ; 
}
void registerBuffer()
{
    if( g_registered ) return ; 
    printf("registerBuffer %d \n", g_buffer_id );
    checkCudaErrors( cudaGraphicsGLRegisterBuffer(&g_graphics_resource, g_buffer_id, g_flags) );
    g_registered = true ; 
}
void mapResources()
{
    checkCudaErrors( cudaGraphicsMapResources(1, &g_graphics_resource, g_stream) );
}
void getMappedPointer() 
{
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer((void **)&g_dev_ptr, &g_bufsize, g_graphics_resource) );
    printf("getMappedPointer g_bufsize %lu g_dev_ptr %p \n", g_bufsize, g_dev_ptr );
}
void unmapResources()
{
    checkCudaErrors( cudaGraphicsUnmapResources(1, &g_graphics_resource, g_stream));
    g_dev_ptr = NULL ;
}

CUdeviceptr mapGLBuffer(unsigned int buffer_id)
{
    setBufferId(buffer_id);
    registerBuffer();
    mapResources();
    getMappedPointer();

    CUdeviceptr cu_ptr = reinterpret_cast<CUdeviceptr>(g_dev_ptr) ; 
    return cu_ptr ;
}
void unmapGLBuffer()
{
    unmapResources();
}


void GLOptiXThrust::referenceBufferForCUDA()
{
    printf("referenceBufferForCUDA %d\n", m_buffer_id);

    CUdeviceptr cu_ptr = mapGLBuffer(m_buffer_id);
    
    m_buffer = m_context->createBufferForCUDA(m_type, m_format, m_size);
    m_buffer->setDevicePointer(m_device, cu_ptr );
}

void GLOptiXThrust::unreferenceBufferForCUDA()
{
    printf("unreferenceBufferForCUDA\n");
    unmapGLBuffer();
    //m_buffer->markDirty();
}



struct scale
{
    float m_factor ; 

    scale(float factor) : m_factor(factor) {}

    __host__ __device__ float4 operator()(float4& v)
    {   
        return make_float4( v.x * m_factor, v.y * m_factor, v.z * m_factor, v.w  ) ;  // not scaling w 
        //return make_float4( 0.5f, 0.5f, 0.5f, 1.0f  ) ;  // not scaling w 
    }   
};



template <typename T>
T* GLOptiXThrust::getRawPointer()
{
    return getRawPointer<T>(m_interop);
}
 

template <typename T>
T* GLOptiXThrust::getRawPointer(Interop_t interop)
{
    T* d_ptr = NULL; 

    CUdeviceptr cu_ptr ; 

    switch(interop)
    {
        case  OCT:
        case GOCT:
                   cu_ptr = m_buffer->getDevicePointer( m_device );  
                   d_ptr = (T*)cu_ptr ;
                   break ;
        case  GCT:
        case GCOT:
                   d_ptr = (T*)g_dev_ptr ;
                   break ;
    }
    return d_ptr ; 
} 


void GLOptiXThrust::postprocess(float factor)
{
    float4* d_ptr = getRawPointer<float4>();

    thrust::device_ptr<float4> dptr = thrust::device_pointer_cast(d_ptr);

    thrust::device_vector<float4> dvec(dptr, dptr+m_size);

    printf("GLOptiXThrust::postprocess scale by factor %10.4f \n", factor);
    scale f(factor);

    thrust::transform( dvec.begin(), dvec.end(), dvec.begin(), f );

    float4 a = dvec[0] ;
    printf("dvec[0] %10.4f %10.4f %10.4f %10.4f \n", a.x, a.y, a.z, a.w );
}

void GLOptiXThrust::sync()
{
    printf("GLOptiXThrust::sync\n");
    //cudaDeviceSynchronize();
    cudaThreadSynchronize();
}



