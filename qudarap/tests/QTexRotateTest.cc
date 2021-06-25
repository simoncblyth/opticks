#include "QTex.hh"

#include <vector_types.h>
#include <vector_functions.h>
#include <cuda_runtime.h>

#include "cudaCheckErrors.h"

#include <iostream>
#define SIMG_IMPLEMENTATION 1 
#include "SIMG.hh"


int main(int argc, char** argv)
{
    const char* ipath = argc > 1 ? argv[1] : "/tmp/i.png" ;
    const char* opath = argc > 2 ? argv[2] : "/tmp/o.png" ;

    int desired_channels = 4 ;
    // hmm *desired_channels* does not change channels, the input image must be 4-channel 
    // (png are often 4-channel, jpg are 3 channel) 

    SIMG img(ipath, desired_channels);
    std::cout << img.desc() << std::endl ;
    assert( img.channels == 4 );

    QTex<uchar4> qtex(img.width, img.height, img.data);

    float theta = 1.f ; // radian
    qtex.rotate(theta); 
    cudaDeviceSynchronize();

    std::cout << "writing to " << opath << std::endl ;

    SIMG img2(img.width, img.height, img.channels, (unsigned char*)qtex.dst );
    img2.writePNG(opath);

    return 0;
}
