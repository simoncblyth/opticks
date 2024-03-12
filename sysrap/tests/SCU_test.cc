/**
SCU_test.cc
============

::

    ~/o/sysrap/tests/SCU_test.sh 


**/
#include "SCU.h"
#include <iostream>

int main()
{
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 

    int32_t width = 1024 ; 
    int32_t height = 768 ; 
    SCU::ConfigureLaunch2D(numBlocks, threadsPerBlock, width, height );   

    std::cout 
        << " width " << width
        << " height " << height
        << " numBlocks " << numBlocks   
        << " threadsPerBlock " << threadsPerBlock
        << std::endl
        ;

    return 0 ; 
}
