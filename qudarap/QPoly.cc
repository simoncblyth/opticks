
#include "PLOG.hh"
#include "scuda.h"

#include "QUDA_CHECK.h"
#include "QPoly.hh"
#include <cuda_runtime.h>


QPoly::QPoly()
{
}


extern "C" void QPoly_demo(dim3 numBlocks, dim3 threadsPerBlock) ; 
extern "C" void QPoly_tmpl_demo(dim3 numBlocks, dim3 threadsPerBlock) ; 


void QPoly::configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height )
{
    threadsPerBlock.x = height == 1 && width < 512 ? width : 512 ; 
    threadsPerBlock.y = 1 ; 
    threadsPerBlock.z = 1 ; 
 
    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ; 
}

void QPoly::demo()
{
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, 1, 1 ); 
    QPoly_demo(numBlocks, threadsPerBlock);  

    cudaDeviceSynchronize();
}

void QPoly::tmpl_demo()
{
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, 1, 1 ); 

    QPoly_tmpl_demo(numBlocks, threadsPerBlock);  

    cudaDeviceSynchronize();
}



