/**
qsimCPUTest.cc
================

Testing GPU code on CPU requires mocking 
of CUDA API, especially tex2D lookups 
and curand random generation. 

There are now lots of examples of curand mocking, 
search for MOCK_CURAND, MOCK_CUDA. See::

    sysrap/s_mock_curand.h 
    sysrap/scurand.h 

Mocking tex2D lookups is not so common. See::

    sysrap/s_mock_texture.h 
    sysrap/stexture.h 

and search for MOCK_TEXTURE, MOCK_CUDA. 

**/

#include "scuda.h"
#include "squad.h"

template<typename T>
T tex2D( cudaTextureObject_t tex, float x, float y )
{  
    float4 lookup = make_float4( 0.f, 0.f, 0.f, 0.f ); 
    return lookup ; 
}

#include "qsim.h"


int main(int argc, char** argv)
{
    qsim<float> qs ; 

    return 0 ; 
}
