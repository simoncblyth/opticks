
#include <stdio.h>
#include "qpoly.h"

__global__ void _QPoly_demo()
{
    RectangleV1 r1 ; r1.set_param(10.0, 10.0) ; 
    RectangleV2 r2 ; r2.set_param(10.0, 10.0) ; 
    TriangleV1 t1  ; t1.set_param(10.0, 10.0) ; 
    TriangleV2 t2  ; t2.set_param(10.0, 10.0) ; 

    printf(" r1.area %10.3f  r2.area %10.3f t1.area %10.3f t2.area %10.3f \n", r1.area(), r2.area(), t1.area(), t2.area() );   
}

extern "C" void QPoly_demo(dim3 numBlocks, dim3 threadsPerBlock ) 
{
    _QPoly_demo<<<numBlocks,threadsPerBlock>>>();
} 


template <typename R, typename T>
 __global__ void _QPoly_tmpl_demo()
{
    R rtmpl ; 
    rtmpl.set_param(10.0, 10.0) ; 

    T ttmpl ; 
    ttmpl.set_param(10.0, 10.0) ; 

    printf(" rtmpl.area %10.3f  ttmpl.area %10.3f \n", rtmpl.area(), ttmpl.area() );   
}

/**
Looks good, BUT: would not work with optix7 extern "C" __raygen_rg functions ?

**/
extern "C" void QPoly_tmpl_demo(dim3 numBlocks, dim3 threadsPerBlock ) 
{
    _QPoly_tmpl_demo<RectangleV1, TriangleV1><<<numBlocks,threadsPerBlock>>>();
    _QPoly_tmpl_demo<RectangleV1, TriangleV2><<<numBlocks,threadsPerBlock>>>();
    _QPoly_tmpl_demo<RectangleV2, TriangleV1><<<numBlocks,threadsPerBlock>>>();
    _QPoly_tmpl_demo<RectangleV2, TriangleV2><<<numBlocks,threadsPerBlock>>>();
} 


