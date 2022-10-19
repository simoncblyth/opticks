#include <optix.h>

/**



**/



extern "C" __global__ void __raygen__rg()
{
    float f0 = 1.f ; 

    // its unclear where these methods are coming from
    // there are __float_as_uint __uint_as_float documented

#ifdef NOW_DEPRECATED_WARNS
    unsigned u0 = float_as_uint( f0 ); 
    float f1 = uint_as_float(u0) ; 
#else
    unsigned u0 = __float_as_uint( f0 ); 
    float f1 = __uint_as_float(u0) ; 
#endif

    //int i1 = uint_as_int(u0);  nope
}

/**
Compiler informs::

/data3/wenzel/newopticks_dev2/opticks/CSGOptiX/Check.cu(11): warning #1444-D: function "float_as_uint"
/usr/local/cuda/include/crt/device_functions.hpp(140): here was declared deprecated ("float_as_uint() is deprecated in favor of __float_as_uint() and may be removed in a fu
/data3/wenzel/newopticks_dev2/opticks/CSGOptiX/Check.cu(12): warning #1444-D: function "uint_as_float"
/usr/local/cuda/include/crt/device_functions.hpp(145): here was declared deprecated ("uint_as_float() is deprecated in favor of __uint_as_float() and may be removed in a fu



**/
