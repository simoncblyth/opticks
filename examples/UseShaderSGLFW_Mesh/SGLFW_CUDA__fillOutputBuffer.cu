#include <stdio.h>
#include <vector_functions.h>

__global__ void _SGLFW_CUDA__fillOutputBuffer( uchar4* output_buffer, int32_t width, int32_t height )
{
    int32_t ix = blockIdx.x*blockDim.x + threadIdx.x;
    int32_t iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    
    int32_t index = iy * width + ix ;  

    //output_buffer[index] = make_uchar4( 0x00, 0xff, 0x00, 0xff );   // green 
    //output_buffer[index] = make_uchar4( 0xff, 0x00, 0x00, 0xff );   // red
    output_buffer[index] = make_uchar4( 0x00, 0x00, 0xff, 0xff );   // blue
}

extern void SGLFW_CUDA__fillOutputBuffer( dim3 numBlocks, dim3 threadsPerBlock, uchar4* d_output_buffer, int32_t width, int32_t height )
{
    printf("//SGLFW_CUDA__fillOutputBuffer width %d height %d \n", width, height );
    _SGLFW_CUDA__fillOutputBuffer<<<numBlocks,threadsPerBlock>>>( d_output_buffer, width, height );
}


