#pragma once
/**
SCUDAOutputBuffer.h
=====================

Adapted from SDK/CUDAOutputBuffer.h
Include this after OpenGL headers.

**/

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "CUDA_CHECK.h"

#include <iostream>
#include <vector>


enum class SCUDAOutputBufferType
{
    CUDA_DEVICE = 0, // not preferred, typically slower than ZERO_COPY
    GL_INTEROP  = 1, // single device only, preferred for single device
    ZERO_COPY   = 2, // general case, preferred for multi-gpu if not fully nvlink connected
    CUDA_P2P    = 3  // fully connected only, preferred for fully nvlink connected
};

typedef int CUstream ;  // kludge for compile with ancient CUDA

template <typename PIXEL_FORMAT>
class SCUDAOutputBuffer
{
public:
    SCUDAOutputBuffer( SCUDAOutputBufferType type, int32_t width, int32_t height );
    ~SCUDAOutputBuffer();

    void setDevice( int32_t device_idx ) { m_device_idx = device_idx; }
    void setStream( CUstream stream    ) { m_stream     = stream;     }

    void resize( int32_t width, int32_t height );

    // Allocate or update device pointer as necessary for CUDA access
    PIXEL_FORMAT* map();
    void unmap();

    int32_t        width() const  { return m_width;  }
    int32_t        height() const { return m_height; }

    // Get output buffer
    GLuint         getPBO();
    void           deletePBO();
    PIXEL_FORMAT*  getHostPointer();

private:
    void makeCurrent() { CUDA_CHECK( cudaSetDevice( m_device_idx ) ); }

    SCUDAOutputBufferType       m_type;

    int32_t                    m_width             = 0u;
    int32_t                    m_height            = 0u;

    cudaGraphicsResource*      m_cuda_gfx_resource = nullptr;
    GLuint                     m_pbo               = 0u;
    PIXEL_FORMAT*              m_device_pixels     = nullptr;
    PIXEL_FORMAT*              m_host_zcopy_pixels = nullptr;
    std::vector<PIXEL_FORMAT>  m_host_pixels;

    CUstream                   m_stream            = 0u;
    int32_t                    m_device_idx        = 0;
};


template <typename PIXEL_FORMAT>
inline SCUDAOutputBuffer<PIXEL_FORMAT>::SCUDAOutputBuffer( SCUDAOutputBufferType type, int32_t width, int32_t height )
    : m_type( type )
{
    if( width < 1 ) width = 1 ; 
    if( height < 1 ) height = 1 ; 

    // If using GL Interop, expect that the active device is also the display device.
    if( type == SCUDAOutputBufferType::GL_INTEROP )
    {
        int current_device, is_display_device;
        CUDA_CHECK( cudaGetDevice( &current_device ) );
        CUDA_CHECK( cudaDeviceGetAttribute( &is_display_device, cudaDevAttrKernelExecTimeout, current_device ) );
        if( !is_display_device ) std::cerr << "GL interop is only available on display device" << std::endl ; 
        assert( is_display_device ); 
    }
    resize( width, height );
}


template <typename PIXEL_FORMAT>
SCUDAOutputBuffer<PIXEL_FORMAT>::~SCUDAOutputBuffer()
{
    try
    {
        makeCurrent();
        if( m_type == SCUDAOutputBufferType::CUDA_DEVICE || m_type == SCUDAOutputBufferType::CUDA_P2P )
        {
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_device_pixels ) ) );
        }
        else if( m_type == SCUDAOutputBufferType::ZERO_COPY )
        {
            CUDA_CHECK( cudaFreeHost( reinterpret_cast<void*>( m_host_zcopy_pixels ) ) );
        }
        else if( m_type == SCUDAOutputBufferType::GL_INTEROP || m_type == SCUDAOutputBufferType::CUDA_P2P )
        {
            CUDA_CHECK( cudaGraphicsUnregisterResource( m_cuda_gfx_resource ) );
        }

        if( m_pbo != 0u )
        {
            GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) );
            GL_CHECK( glDeleteBuffers( 1, &m_pbo ) );
        }
    }
    catch(std::exception& e )
    {
        std::cerr << "SCUDAOutputBuffer destructor caught exception: " << e.what() << std::endl;
    }
}


template <typename PIXEL_FORMAT>
void SCUDAOutputBuffer<PIXEL_FORMAT>::resize( int32_t width, int32_t height )
{
    if( width < 1 ) width = 1 ; 
    if( height < 1 ) height = 1 ; 

    if( m_width == width && m_height == height )
        return;

    m_width  = width;
    m_height = height;

    makeCurrent();

    if( m_type == SCUDAOutputBufferType::CUDA_DEVICE || m_type == SCUDAOutputBufferType::CUDA_P2P )
    {
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_device_pixels ) ) );
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &m_device_pixels ),
                    m_width*m_height*sizeof(PIXEL_FORMAT)
                    ) );

    }

    if( m_type == SCUDAOutputBufferType::GL_INTEROP || m_type == SCUDAOutputBufferType::CUDA_P2P )
    {
        // GL buffer gets resized below
        GL_CHECK( glGenBuffers( 1, &m_pbo ) );
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, m_pbo ) );
        GL_CHECK( glBufferData( GL_ARRAY_BUFFER, sizeof(PIXEL_FORMAT)*m_width*m_height, nullptr, GL_STREAM_DRAW ) );
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0u ) );

        CUDA_CHECK( cudaGraphicsGLRegisterBuffer(
                    &m_cuda_gfx_resource,
                    m_pbo,
                    cudaGraphicsMapFlagsWriteDiscard
                    ) );
    }

    if( m_type == SCUDAOutputBufferType::ZERO_COPY )
    {
        CUDA_CHECK( cudaFreeHost( reinterpret_cast<void*>( m_host_zcopy_pixels ) ) );
        CUDA_CHECK( cudaHostAlloc(
                    reinterpret_cast<void**>( &m_host_zcopy_pixels ),
                    m_width*m_height*sizeof(PIXEL_FORMAT),
                    cudaHostAllocPortable | cudaHostAllocMapped
                    ) );
        CUDA_CHECK( cudaHostGetDevicePointer(
                    reinterpret_cast<void**>( &m_device_pixels ),
                    reinterpret_cast<void*>( m_host_zcopy_pixels ),
                    0 /*flags*/
                    ) );
    }

    if( m_type != SCUDAOutputBufferType::GL_INTEROP && m_type != SCUDAOutputBufferType::CUDA_P2P && m_pbo != 0u )
    {
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, m_pbo ) );
        GL_CHECK( glBufferData( GL_ARRAY_BUFFER, sizeof(PIXEL_FORMAT)*m_width*m_height, nullptr, GL_STREAM_DRAW ) );
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0u ) );
    }

    if( !m_host_pixels.empty() )
        m_host_pixels.resize( m_width*m_height );
}


template <typename PIXEL_FORMAT>
PIXEL_FORMAT* SCUDAOutputBuffer<PIXEL_FORMAT>::map()
{
    if( m_type == SCUDAOutputBufferType::CUDA_DEVICE || m_type == SCUDAOutputBufferType::CUDA_P2P )
    {
        // nothing needed
    }
    else if( m_type == SCUDAOutputBufferType::GL_INTEROP  )
    {
        makeCurrent();

        size_t buffer_size = 0u;
        CUDA_CHECK( cudaGraphicsMapResources ( 1, &m_cuda_gfx_resource, m_stream ) );
        CUDA_CHECK( cudaGraphicsResourceGetMappedPointer(
                    reinterpret_cast<void**>( &m_device_pixels ),
                    &buffer_size,
                    m_cuda_gfx_resource
                    ) );
    }
    else // m_type == SCUDAOutputBufferType::ZERO_COPY
    {
        // nothing needed
    }

    return m_device_pixels;
}


template <typename PIXEL_FORMAT>
void SCUDAOutputBuffer<PIXEL_FORMAT>::unmap()
{
    makeCurrent();

    if( m_type == SCUDAOutputBufferType::CUDA_DEVICE || m_type == SCUDAOutputBufferType::CUDA_P2P )
    {
        CUDA_CHECK( cudaStreamSynchronize( m_stream ) );
    }
    else if( m_type == SCUDAOutputBufferType::GL_INTEROP  )
    {
        CUDA_CHECK( cudaGraphicsUnmapResources ( 1, &m_cuda_gfx_resource,  m_stream ) );
    }
    else // m_type == SCUDAOutputBufferType::ZERO_COPY
    {
        CUDA_CHECK( cudaStreamSynchronize( m_stream ) );
    }
}


template <typename PIXEL_FORMAT>
GLuint SCUDAOutputBuffer<PIXEL_FORMAT>::getPBO()
{
    if( m_pbo == 0u )
        GL_CHECK( glGenBuffers( 1, &m_pbo ) );

    const size_t buffer_size = m_width*m_height*sizeof(PIXEL_FORMAT);

    if( m_type == SCUDAOutputBufferType::CUDA_DEVICE )
    {
        // We need a host buffer to act as a way-station
        if( m_host_pixels.empty() )
            m_host_pixels.resize( m_width*m_height );

        makeCurrent();
        CUDA_CHECK( cudaMemcpy(
                    static_cast<void*>( m_host_pixels.data() ),
                    m_device_pixels,
                    buffer_size,
                    cudaMemcpyDeviceToHost
                    ) );

        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, m_pbo ) );
        GL_CHECK( glBufferData(
                    GL_ARRAY_BUFFER,
                    buffer_size,
                    static_cast<void*>( m_host_pixels.data() ),
                    GL_STREAM_DRAW
                    ) );
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) );
    }
    else if( m_type == SCUDAOutputBufferType::GL_INTEROP  )
    {
        // Nothing needed
    }
    else if ( m_type == SCUDAOutputBufferType::CUDA_P2P )
    {
        makeCurrent();
        void* pbo_buff = nullptr;
        size_t dummy_size = 0;

        CUDA_CHECK( cudaGraphicsMapResources( 1, &m_cuda_gfx_resource, m_stream ) );
        CUDA_CHECK( cudaGraphicsResourceGetMappedPointer( &pbo_buff, &dummy_size, m_cuda_gfx_resource ) );
        CUDA_CHECK( cudaMemcpy( pbo_buff, m_device_pixels, buffer_size, cudaMemcpyDeviceToDevice ) );
        CUDA_CHECK( cudaGraphicsUnmapResources( 1, &m_cuda_gfx_resource, m_stream ) );
    }
    else // m_type == SCUDAOutputBufferType::ZERO_COPY
    {
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, m_pbo ) );
        GL_CHECK( glBufferData(
                    GL_ARRAY_BUFFER,
                    buffer_size,
                    static_cast<void*>( m_host_zcopy_pixels ),
                    GL_STREAM_DRAW
                    ) );
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) );
    }

    return m_pbo;
}

template <typename PIXEL_FORMAT>
void SCUDAOutputBuffer<PIXEL_FORMAT>::deletePBO()
{
    GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) );
    GL_CHECK( glDeleteBuffers( 1, &m_pbo ) );
    m_pbo = 0;
}

template <typename PIXEL_FORMAT>
PIXEL_FORMAT* SCUDAOutputBuffer<PIXEL_FORMAT>::getHostPointer()
{
    if( m_type == SCUDAOutputBufferType::CUDA_DEVICE ||
        m_type == SCUDAOutputBufferType::CUDA_P2P ||
        m_type == SCUDAOutputBufferType::GL_INTEROP  )
    {
        m_host_pixels.resize( m_width*m_height );

        makeCurrent();
        CUDA_CHECK( cudaMemcpy(
                    static_cast<void*>( m_host_pixels.data() ),
                    map(),
                    m_width*m_height*sizeof(PIXEL_FORMAT),
                    cudaMemcpyDeviceToHost
                    ) );
        unmap();

        return m_host_pixels.data();
    }
    else // m_type == SCUDAOutputBufferType::ZERO_COPY
    {
        return m_host_zcopy_pixels;
    }
}

