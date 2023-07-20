CSGOptiX_C4_build_issue_missing_WITH_THRUST
==============================================

Loads of errors from C4 compilation. Seen these before when WITH_THRUST
not enabled.  

::

    icks/head/build/CSGOptiX/CSGOptiX_generated_Check.cu.ptx
    -- bcm_auto_pkgconfig_each LIB:Threads::Threads : MISSING LIB_PKGCONFIG_NAME 
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /data/blyth/junotop/ExternalLibs/opticks/head/build/CSGOptiX
    [  2%] Building NVCC ptx file CSGOptiX_generated_CSGOptiX7.cu.ptx
    /data/blyth/junotop/ExternalLibs/custom4/0.1.5/include/Custom4/C4MultiLayrStack.h(588): warning: calling a constexpr __host__ function("complex") from a __host__ __device__ function("calc") is not allowed. The experimental flag '--expt-relaxed-constexpr' can be used to allow this.
              detected during:
                instantiation of "void Stack<F, N>::calc(F, F, F, const F *, unsigned int) [with F=float, N=4]" 
    /data/blyth/junotop/ExternalLibs/opticks/head/include/QUDARap/qpmt.h(329): here
                instantiation of "void qpmt<F>::get_lpmtid_ARTE(F *, int, F, F, F) const [with F=float]" 
    /data/blyth/junotop/ExternalLibs/opticks/head/include/QUDARap/qsim.h(1488): here

    /data/blyth/junotop/ExternalLibs/custom4/0.1.5/include/Custom4/C4MultiLayrStack.h(589): warning: calling a constexpr __host__ function("complex") from a __host__ __device__ function("calc") is not allowed. The experimental flag '--expt-relaxed-constexpr' can be used to allow this.
              detected during:
                instantiation of "void Stack<F, N>::calc(F, F, F, const F *, unsigned int) [with F=float, N=4]" 
    /data/blyth/junotop/ExternalLibs/opticks/head/include/QUDARap/qpmt.h(329): here
                instantiation of "void qpmt<F>::get_lpmtid_ARTE(F *, int, F, F, F) const [with F=float]" 
    /data/blyth/junotop/ExternalLibs/opticks/head/include/QUDARap/qsim.h(1488): here

    /data/blyth/junotop/ExternalLibs/custom4/0.1.5/include/Custom4/C4MultiLayrStack.h(150): warning: calling a __host__ function from a __host__ __device__ function is not allowed
              detected during:
                instantiation of "void Matx<T>::reset() [with T=float]" 


::

    epsilon:opticks blyth$ opticks-f WITH_THRUST
    ./sysrap/stmm.h:#ifdef WITH_THRUST
    ./sysrap/stmm.h:    #ifdef WITH_THRUST
    ./sysrap/stmm.h:#ifdef WITH_THRUST
    ./sysrap/stmm.h:#ifdef WITH_THRUST
    ./sysrap/stmm.h:#ifdef WITH_THRUST 
    ./sysrap/stmm.h:#ifdef WITH_THRUST 
    ./sysrap/stmm.h:#ifdef WITH_THRUST
    ./qudarap/CMakeLists.txt:WITH_THRUST compile definition
    ./qudarap/CMakeLists.txt:* WITH_THRUST is needed for Custom4 on GPU complex math functions used by qpmt.h 
    ./qudarap/CMakeLists.txt:* However Custom4 also used on CPU without WITH_THRUST (eg by junosw), so 
    ./qudarap/CMakeLists.txt:  the WITH_THRUST definition must be PRIVATE to avoid interference with other
    ./qudarap/CMakeLists.txt:target_compile_definitions( ${name} PRIVATE WITH_THRUST )
    ./qudarap/tests/QPMT_Test.sh:             -DWITH_THRUST \
    ./qudarap/tests/QPMT_Test.sh:        -DWITH_THRUST \
    ./qudarap/tests/QPMT_Test.sh:        -DWITH_THRUST \
    ./examples/UseThrust/basic_complex.h:#ifdef WITH_THRUST
    ./examples/UseThrust/basic_complex.h:#ifdef WITH_THRUST
    ./examples/UseThrust/basic_complex.sh:            -DWITH_THRUST -o /tmp/$name 
    ./examples/UseThrust/basic_complex.sh:            -DWITH_THRUST -o /tmp/$hname
    ./examples/UseThrust/basic_complex.cu:#ifdef WITH_THRUST
    ./examples/UseThrust/basic_complex.cu:    printf("on_device launch WITH_THRUST\n"); 
    ./examples/UseThrust/basic_complex.cu:#ifdef WITH_THRUST
    ./examples/UseThrust/basic_complex.cu:    printf("on_host WITH_THRUST\n"); 
    ./examples/UseThrust/basic_complex.cu:    printf("on_host NOT WITH_THRUST\n"); 
    ./examples/UseThrust/basic_complex_host.cc:#ifdef WITH_THRUST
    ./examples/UseThrust/basic_complex_host.cc:    printf("on_host WITH_THRUST\n"); 
    ./examples/UseThrust/basic_complex_host.cc:    printf("on_host NOT WITH_THRUST\n"); 
    epsilon:opticks blyth$ 


qudarap/CMakeLists.txt::

    185 #[=[
    186 
    187 WITH_THRUST compile definition
    188 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    189 
    190 * WITH_THRUST is needed for Custom4 on GPU complex math functions used by qpmt.h 
    191 
    192 * However Custom4 also used on CPU without WITH_THRUST (eg by junosw), so 
    193   the WITH_THRUST definition must be PRIVATE to avoid interference with other
    194   Custom4 usage
    195 
    196 #]=]
    197 
    198 target_compile_definitions( ${name} PRIVATE WITH_THRUST )



Seems fixed by adding the "PRIVATE WITH_THRUST" to CSGOptiX/CMakeLists.txt

