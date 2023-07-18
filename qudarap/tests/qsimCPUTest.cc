/**
qsimCPUTest.cc
================

Testing GPU code on CPU requires mocking 
of CUDA API, especially tex2D lookups 
and curand random generation. 

There are now lots of examples of curand mocking, search for MOCK_CURAND. 
The initial expts are in sysrap/s_mock_curand.h 

Mocking tex2D lookups is not so common.
Some expts in sysrap/s_mock_texture.h 

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
