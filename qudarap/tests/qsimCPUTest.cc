/**
qsimCPUTest.cc
================

Testing GPU code on CPU is a stretch, would need to 
mock tex2D and curand.

More realistic is testing QUDARap code intended to be 
used with OptiX 7 with CUDA alone.  That is 
what many of the QUDARap tests do.

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
