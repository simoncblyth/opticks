#include <optix.h>


extern "C" __global__ void __raygen__rg()
{
    float f0 = 1.f ; 

    // its unclear where these methods are coming from
    // there are __float_as_uint __uint_as_float documented

    unsigned u0 = float_as_uint( f0 ); 
    float f1 = uint_as_float(u0) ; 

    //int i1 = uint_as_int(u0);  nope



}


