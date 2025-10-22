cuda13_thrust_unary_function
===============================

Report from Nicola::

    About the compilation error: here's what I get:

    [ 19%] Building CXX object CMakeFiles/QUDARap.dir/QTexRotate.cc.o
    /home/mori/software/install/opticks/include/SysRap/strided_range.h(65):
    error: namespace "thrust" has no member class "unary_function"
           struct stride_functor : public
    thrust::unary_function<difference_type,difference_type>
                                                  ^

    plus a lot of other lines. It seems that thrust::unary_function has been
    deprecated in CUDA 12.8 (e.g. see here:
    https://github.com/microsoft/onnxruntime/issues/23499) so it might have
    been removed in CUDA 13 which I'm using, and hence the error. So I'd say
    that CUDA < 13 is needed for Opticks. Unfortunately getting that on my
    system (Archlinux) is not that easy, maybe the master version of Opticks
    has been already fixed for this?
    Cheers,

    Nicola




::

    (ok) A[blyth@localhost sysrap]$ grep unary_function *.*
    strided_range.h:    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    (ok) A[blyth@localhost sysrap]$ grep binary_function *.*
    (ok) A[blyth@localhost sysrap]$ opticks-f thrust::unary_function
    ./examples/UseCUDARapThrust/UseCUDARapThrust.cu:    public thrust::unary_function<unsigned int, float> 
    ./sysrap/strided_range.h:    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    ./thrustrap/TIsHit.hh:struct TIsHit4x4 : public thrust::unary_function<float4x4,bool>
    ./thrustrap/TIsHit.hh:struct TIsHit2x4 : public thrust::unary_function<float2x4,bool>
    ./thrustrap/TIsHit.hh:struct TIsHit : public thrust::unary_function<T,bool>
    ./thrustrap/TSparse_.cu:struct apply_lookup_functor : public thrust::unary_function<T,S>
    ./thrustrap/repeated_range.h:    struct repeat_functor : public thrust::unary_function<difference_type,difference_type>
    ./thrustrap/strided_repeated_range.h:    struct stride_repeat_functor : public thrust::unary_function<difference_type,difference_type>
    (ok) A[blyth@localhost opticks]$ 
    (ok) A[blyth@localhost opticks]$ 



