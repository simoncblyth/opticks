optixir
========


::

    lo] A[blyth@localhost include]$ opticks-f optixir
    ./sysrap/tests/SGLFW_SOPTIX_Scene_test_local.sh:xir=$FOLD/SOPTIX.optixir
    ./sysrap/SOPTIX_Module.h:WIP : Generalize this to work with optixir binary as well as PTX text
    ./sysrap/SOPTIX_Module.h:also does optixir ?
    ./sysrap/SOPTIX_Module.h:    else if( sstr::EndsWith(path, "optixir")) type = OPTIXIR ;
    [lo] A[blyth@localhost opticks]$ 



SOPTIX_Module.h not yet used by CSGOptiX
----------------------------------------

::

    [lo] A[blyth@localhost CSGOptiX]$ opticks-f SOPTIX_Module.h
    ./sysrap/SOPTIX.h:#include "SOPTIX_Module.h"
    ./sysrap/tests/SOPTIX_Module_test.cc:#include "SOPTIX_Module.h"
    ./sysrap/tests/SOPTIX_Pipeline_test.cc:#include "SOPTIX_Module.h"
    ./sysrap/tests/SOPTIX_Scene_test.cc:#include "SOPTIX_Module.h"
    ./sysrap/SOPTIX_Module.h:SOPTIX_Module.h : Create OptixModule loaded from file
    ./sysrap/CMakeLists.txt:    SOPTIX_Module.h
    [lo] A[blyth@localhost opticks]$ 



optixModuleCreate
-------------------

::

    [lo] A[blyth@localhost opticks]$ opticks-f optixModuleCreate
    ./CSGOptiX/PIP.cc:    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
    ./CSGOptiX/PIP.cc:    OPTIX_CHECK_LOG( optixModuleCreate(
    ./examples/UseOptiX7GeometryInstanced/PIP.cc:    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
    ./examples/UseOptiX7GeometryInstancedGAS/PIP.cc:    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
    ./examples/UseOptiX7GeometryInstancedGASComp/PIP.cc:    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
    ./examples/UseOptiX7GeometryInstancedGASCompDyn/PIP.cc:    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
    ./examples/UseOptiX7GeometryModular/PIP.cc:    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
    ./examples/UseOptiX7GeometryStandalone/UseOptiX7GeometryStandalone.cc:        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
    ./externals/optix7sdk.bash:optixModuleCreateAbort
    ./sysrap/SOPTIX_Module.h:HMM: in OptiX 7.5 seems the optixModuleCreateFromPTX
    ./sysrap/SOPTIX_Module.h:    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
    ./sysrap/SOPTIX_Module.h:    OPTIX_CHECK_LOG( optixModuleCreate(
    ./sysrap/SOPTIX_Module.h:    if(sizeof_log > 0) std::cout << "[optixModuleCreate _log \n" << _log  << "\n]optixModuleCreate _log\n" ;
    [lo] A[blyth@localhost opticks]$ 


Change sysrap to use optixir
------------------------------

Release build warning::

    nvcc warning : incompatible redefinition for option 'optimize', the last value of this option was used 

::

    [lo] A[blyth@localhost sysrap]$ VERBOSE=1 make | grep nvcc
    /usr/local/cuda-13.1/bin/nvcc -forward-unknown-to-host-compiler  --options-file CMakeFiles/SysRap_OPTIX.dir/includes_CUDA.rsp --expt-extended-lambda -O3 -DNDEBUG -std=c++17 "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -Xcompiler=-fPIC -O2 --use_fast_math -Xcompiler -fPIC -MD -MT CMakeFiles/SysRap_OPTIX.dir/SOPTIX.optixir -MF CMakeFiles/SysRap_OPTIX.dir/SOPTIX.optixir.d -x cu -rdc=true -optix-ir /home/blyth/opticks/sysrap/SOPTIX.cu -o CMakeFiles/SysRap_OPTIX.dir/SOPTIX.optixir
    nvcc warning : incompatible redefinition for option 'optimize', the last value of this option was used
    [lo] A[blyth@localhost sysrap]$ 




int128 issue with optixir compile
------------------------------------------------


::

    [lo] A[blyth@localhost CSGOptiX]$ ./optixir.sh 
    /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/std/__utility/convert_to_integral.h(69): error: "__int128_t (__int128_t)" (aka "__int128 (__int128)") contains a 128-bit integer, which is not supported in device code
      __attribute__((__visibility__("hidden"))) constexpr __int128_t __convert_to_integral(__int128_t __val)
                                                                     ^

    /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/std/__utility/convert_to_integral.h(74): error: "__uint128_t (__uint128_t)" (aka "unsigned __int128 (unsigned __int128)") contains a 128-bit integer, which is not supported in device code
      __attribute__((__visibility__("hidden"))) constexpr __uint128_t __convert_to_integral(__uint128_t __val)
                                                                      ^

    /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/__cmath/ilog.h(100): error: "cuda::std::__4::array<__uint128_t, 39UL> () noexcept" (aka "cuda::std::__4::array<unsigned __int128, 39UL> () noexcept") contains a 128-bit integer, which is not supported in device code
                   __attribute__((__visibility__("hidden"))) constexpr ::cuda::std::__4::array<__uint128_t, 39> __power_of_10_128bit() noexcept
                                                                                                                ^

    /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/std/array(91): error: "void (const cuda::std::__4::array<__uint128_t, 39UL>::value_type &)" (aka "void (const __uint128_t &)") contains a 128-bit integer, which is not supported in device code
       __attribute__((__visibility__("hidden"))) constexpr void fill(const value_type& __u)
                                                                ^
              detected during instantiation of class "cuda::std::__4::array<_Tp, _Size> [with _Tp=__uint128_t, _Size=39UL]" at line 101 of /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/__cmath/ilog.h

    /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/std/array(96): error: "void (cuda::std::__4::array<__uint128_t, 39UL> &) noexcept(<expr>)" (aka "void (cuda::std::__4::array<unsigned __int128, 39UL> &) noexcept(<expr>)") contains a 128-bit integer, which is not supported in device code
       __attribute__((__visibility__("hidden"))) constexpr void swap(array& __a) noexcept(__is_nothrow_swappable<_Tp>::value)
                                                                ^
              detected during instantiation of class "cuda::std::__4::array<_Tp, _Size> [with _Tp=__uint128_t, _Size=39UL]" at line 101 of /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/__cmath/ilog.h

    /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/std/array(102): error: "cuda::std::__4::array<__uint128_t, 39UL>::iterator () noexcept" (aka "unsigned __int128 *() noexcept") contains a 128-bit integer, which is not supported in device code
                     __attribute__((__visibility__("hidden"))) constexpr iterator begin() noexcept
                                                                                  ^
              detected during instantiation of class "cuda::std::__4::array<_Tp, _Size> [with _Tp=__uint128_t, _Size=39UL]" at line 101 of /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/__cmath/ilog.h

    /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/std/array(106): error: "cuda::std::__4::array<__uint128_t, 39UL>::const_iterator () const noexcept" (aka "const __uint128_t *() const noexcept") contains a 128-bit integer, which is not supported in device code
                     __attribute__((__visibility__("hidden"))) constexpr const_iterator begin() const noexcept
                                                                                        ^
              detected during instantiation of class "cuda::std::__4::array<_Tp, _Size> [with _Tp=__uint128_t, _Size=39UL]" at line 101 of /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/__cmath/ilog.h

    /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/std/array(110): error: "cuda::std::__4::array<__uint128_t, 39UL>::iterator () noexcept" (aka "unsigned __int128 *() noexcept") contains a 128-bit integer, which is not supported in device code
                     __attribute__((__visibility__("hidden"))) constexpr iterator end() noexcept
                                                                                  ^
              detected during instantiation of class "cuda::std::__4::array<_Tp, _Size> [with _Tp=__uint128_t, _Size=39UL]" at line 101 of /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/__cmath/ilog.h

    /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/std/array(114): error: "cuda::std::__4::array<__uint128_t, 39UL>::const_iterator () const noexcept" (aka "const __uint128_t *() const noexcept") contains a 128-bit integer, which is not supported in device code
                     __attribute__((__visibility__("hidden"))) constexpr const_iterator end() const noexcept
                                                                                        ^
              detected during instantiation of class "cuda::std::__4::array<_Tp, _Size> [with _Tp=__uint128_t, _Size=39UL]" at line 101 of /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/__cmath/ilog.h

    /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/std/array(119): error: "cuda::std::__4::array<__uint128_t, 39UL>::reverse_iterator () noexcept" (aka "cuda::std::__4::reverse_iterator<unsigned __int128 *> () noexcept") contains a 128-bit integer, which is not supported in device code
                     __attribute__((__visibility__("hidden"))) constexpr reverse_iterator rbegin() noexcept
                                                                                          ^
              detected during instantiation of class "cuda::std::__4::array<_Tp, _Size> [with _Tp=__uint128_t, _Size=39UL]" at line 101 of /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/__cmath/ilog.h

    /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/std/array(123): error: "cuda::std::__4::array<__uint128_t, 39UL>::const_reverse_iterator () const noexcept" (aka "cuda::std::__4::reverse_iterator<const __uint128_t *> () const noexcept") contains a 128-bit integer, which is not supported in device code
                     __attribute__((__visibility__("hidden"))) constexpr const_reverse_iterator rbegin() const noexcept
                                                                                                ^
              detected during instantiation of class "cuda::std::__4::array<_Tp, _Size> [with _Tp=__uint128_t, _Size=39UL]" at line 101 of /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/__cmath/ilog.h

    /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/std/array(127): error: "cuda::std::__4::array<__uint128_t, 39UL>::reverse_iterator () noexcept" (aka "cuda::std::__4::reverse_iterator<unsigned __int128 *> () noexcept") contains a 128-bit integer, which is not supported in device code
                     __attribute__((__visibility__("hidden"))) constexpr reverse_iterator rend() noexcept
                                                                                          ^
              detected during instantiation of class "cuda::std::__4::array<_Tp, _Size> [with _Tp=__uint128_t, _Size=39UL]" at line 101 of /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/__cmath/ilog.h

    /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/std/array(131): error: "cuda::std::__4::array<__uint128_t, 39UL>::const_reverse_iterator () const noexcept" (aka "cuda::std::__4::reverse_iterator<const __uint128_t *> () const noexcept") contains a 128-bit integer, which is not supported in device code
                     __attribute__((__visibility__("hidden"))) constexpr const_reverse_iterator rend() const noexcept
                                                                                                ^
              detected during instantiation of class "cuda::std::__4::array<_Tp, _Size> [with _Tp=__uint128_t, _Size=39UL]" at line 101 of /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/__cmath/ilog.h

    /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/std/array(136): error: "cuda::std::__4::array<__uint128_t, 39UL>::const_iterator () const noexcept" (aka "const __uint128_t *() const noexcept") contains a 128-bit integer, which is not supported in device code
                     __attribute__((__visibility__("hidden"))) constexpr const_iterator cbegin() const noexcept
                                                                                        ^
              detected during instantiation of class "cuda::std::__4::array<_Tp, _Size> [with _Tp=__uint128_t, _Size=39UL]" at line 101 of /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/__cmath/ilog.h

    /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/std/array(140): error: "cuda::std::__4::array<__uint128_t, 39UL>::const_iterator () const noexcept" (aka "const __uint128_t *() const noexcept") contains a 128-bit integer, which is not supported in device code
                     __attribute__((__visibility__("hidden"))) constexpr const_iterator cend() const noexcept
                                                                                        ^
              detected during instantiation of class "cuda::std::__4::array<_Tp, _Size> [with _Tp=__uint128_t, _Size=39UL]" at line 101 of /usr/local/cuda-13.1/targets/x86_64-linux/include/cccl/cuda/__cmath/ilog.h



Binary search of the headers points to qsim.h / qpmt.h 
----------------------------------------------------------

Drilling down the issue is from::

    #ifdef WITH_CUSTOM4
    #include "C4MultiLayrStack.h"
    #endif

Most likely thrust/complex.h::

    #ifdef WITH_THRUST
    #include <thrust/complex.h>
    #else
    #include <complex>
    #include <cmath>
    #endif

    #if defined(__CUDACC__) || defined(__CUDABE__)
    #else
    #include <iostream>
    #include <iomanip>
    #include <sstream>
    #include <string>
    #include <cassert>
    #include <cstdlib>
    #include <vector>
    #include <array>
    #endif

YEP : confirmed - all those int128 errors are from::

    #include <thrust/complex.h>




