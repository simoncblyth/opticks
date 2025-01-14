#include "QTex.hh"
#include "QTexRotate.hh"

#include <vector_types.h>
#include <vector_functions.h>
#include <cuda_runtime.h>

#include "cudaCheckErrors.h"

#include <iostream>
#define SIMG_IMPLEMENTATION 1 
#include "SIMG.h"

struct NP ; 

int main(int argc, char** argv)
{
    const char* ipath = argc > 1 ? argv[1] : "/tmp/i.png" ;
    const char* opath = argc > 2 ? argv[2] : "/tmp/o.png" ;

    int desired_channels = 4 ;
    // hmm *desired_channels* does not change channels, the input image must be 4-channel 
    // (png are often 4-channel, jpg are 3 channel) 

    SIMG img(ipath, desired_channels);
    std::cout << img.desc() << std::endl ;

    if( img.channels == 0 )
    {
        std::cout 
            << argv[0]
            << " failed to open input path to png image with 4 channels  " << ipath 
            << std::endl 
            ;
        return 0 ; 
    }

    assert( img.channels == 4 );

    char filterMode = 'P' ; // cudaFilterModePoint : no interpolation, necessary with uchar4 
    bool normalizedCoords = false ; 
    const NP* a = nullptr ; 
 
    QTex<uchar4> qtex(img.width, img.height, img.data, filterMode, normalizedCoords, a );


    QTexRotate<uchar4> qrot(&qtex); 

    float theta = 1.f ; // radian
    qrot.rotate(theta); 

    cudaDeviceSynchronize();

    std::cout << "writing to " << opath << std::endl ;

    SIMG img2(img.width, img.height, img.channels, (unsigned char*)qrot.rotate_dst );
    img2.writePNG(opath);

    return 0;
}

