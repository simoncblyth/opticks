#pragma once
/**
SGLFW_CUDA.h : Coordinate SCUDA_OutputBuffer and SGLDisplay for display of interop buffers
===========================================================================================


**/


#include "SCU.h"
#include "SCUDA_OutputBuffer.h"
#include "SGLDisplay.h"

struct SGLFW_CUDA
{
    SGLM& gm ; 
    SCUDA_OutputBuffer<uchar4>* output_buffer ; 
    SGLDisplay* gl_display ; 

    SGLFW_CUDA(SGLM& gm); 
    void init();

    void fillOutputBuffer(); 
    void displayOutputBuffer(GLFWwindow* window);
}; 

inline SGLFW_CUDA::SGLFW_CUDA(SGLM& _gm)
    :
    gm(_gm),
    output_buffer( nullptr ), 
    gl_display( nullptr )
{
    init();
}

inline void SGLFW_CUDA::init()
{
    output_buffer = new SCUDA_OutputBuffer<uchar4>( SCUDA_OutputBufferType::GL_INTEROP, gm.Width(), gm.Height() ) ; 
    //std::cout << "SGLFW_CUDA::init output_buffer.desc " << output_buffer->desc() ; 
    gl_display = new SGLDisplay ; 
    //std::cout << "SGLFW_CUDA::init gl_display.desc " << gl_display->desc() ; 
}


extern void SGLFW_CUDA__fillOutputBuffer( dim3 numBlocks, dim3 threadsPerBlock, uchar4* output_buffer, int width, int height ); 
/**
SGLFW_CUDA::fillOutputBuffer
----------------------------

Unused ? In anycase intended for debug ? 

**/
inline void SGLFW_CUDA::fillOutputBuffer()
{
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    SCU::ConfigureLaunch2D(numBlocks, threadsPerBlock, output_buffer->width(), output_buffer->height() );   

    SGLFW_CUDA__fillOutputBuffer(numBlocks, threadsPerBlock, 
         output_buffer->map(), 
         output_buffer->width(), 
         output_buffer->height() );           

    output_buffer->unmap();
    CUDA_SYNC_CHECK();
}

/**
SGLFW_CUDA::displayOutputBuffer
--------------------------------

Primary method, eg called from render loop of CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc 

This enables OpenGL presentation of GPU buffer writted by CUDA/OptiX

**/

inline void SGLFW_CUDA::displayOutputBuffer(GLFWwindow* window)
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //  
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );

    gl_display->display(
            output_buffer->width(),
            output_buffer->height(),
            framebuf_res_x,
            framebuf_res_y,
            output_buffer->getPBO()
            );  
}

