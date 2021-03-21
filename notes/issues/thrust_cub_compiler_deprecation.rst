thrust_cub_compiler_deprecation
==================================

::

    -- Build files have been written to: /hpcfs/juno/junogpu/blyth/local/opticks/build/thrustrap
    === om-make-one : thrustrap       /hpcfs/juno/junogpu/blyth/opticks/thrustrap                  /hpcfs/juno/junogpu/blyth/local/opticks/build/thrustrap      
    [  8%] Building NVCC (Device) object CMakeFiles/ThrustRap.dir/ThrustRap_generated_TUtil_.cu.o
    [  8%] Building NVCC (Device) object CMakeFiles/ThrustRap.dir/ThrustRap_generated_TBufPair_.cu.o
    [ 11%] Building NVCC (Device) object CMakeFiles/ThrustRap.dir/ThrustRap_generated_TBuf_.cu.o
    [ 11%] Building NVCC (Device) object CMakeFiles/ThrustRap.dir/ThrustRap_generated_TCURANDImp.cu.o
    [ 13%] Building NVCC (Device) object CMakeFiles/ThrustRap.dir/ThrustRap_generated_TRngBuf_.cu.o
    [ 13%] Building NVCC (Device) object CMakeFiles/ThrustRap.dir/ThrustRap_generated_TSparse_.cu.o
    In file included from /usr/local/cuda/include/thrust/detail/config/config.h:27:0,
                     from /usr/local/cuda/include/thrust/detail/config.h:23,
                     from /usr/local/cuda/include/thrust/device_vector.h:24,
                     from /hpcfs/juno/junogpu/blyth/opticks/thrustrap/TUtil.hh:31,
                     from /hpcfs/juno/junogpu/blyth/opticks/thrustrap/TUtil_.cu:20:
    /usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:92:13: warning: Thrust requires GCC 5.0. Please upgrade your compiler. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message. [enabled by default]
          THRUST_COMPILER_DEPRECATION(GCC 5.0, upgrade your compiler);
                 ^
    /usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:104:13: warning: Thrust requires C++14. Please pass -std=c++14 to your compiler. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message. [enabled by default]
       THRUST_COMPILER_DEPRECATION(C++14, pass -std=c++14 to your compiler);
                 ^
    In file included from /usr/local/cuda/include/thrust/detail/config/config.h:27:0,
                     from /usr/local/cuda/include/thrust/detail/config.h:23,
                     from /usr/local/cuda/include/thrust/device_vector.h:24,
                     from /hpcfs/juno/junogpu/blyth/opticks/thrustrap/TBuf_.cu:25:
    /usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:92:13: warning: Thrust requires GCC 5.0. Please upgrade your compiler. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message. [enabled by default]
          THRUST_COMPILER_DEPRECATION(GCC 5.0, upgrade your compiler);
                 ^
    /usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:104:13: warning: Thrust requires C++14. Please pass -std=c++14 to your compiler. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message. [enabled by default]
       THRUST_COMPILER_DEPRECATION(C++14, pass -std=c++14 to your compiler);
                 ^
    In file included from /usr/local/cuda/include/thrust/detail/config/config.h:27:0,
                     from /usr/local/cuda/include/thrust/detail/config.h:23,
                     from /usr/local/cuda/include/thrust/for_each.h:23,
                     from /hpcfs/juno/junogpu/blyth/opticks/thrustrap/TRngBuf_.cu:23:
    /usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:92:13: warning: Thrust requires GCC 5.0. Please upgrade your compiler. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message. [enabled by default]
          THRUST_COMPILER_DEPRECATION(GCC 5.0, upgrade your compiler);
                 ^
    /usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:104:13: warning: Thrust requires C++14. Please pass -std=c++14 to your compiler. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message. [enabled by default]
       THRUST_COMPILER_DEPRECATION(C++14, pass -std=c++14 to your compiler);
                 ^
    In file included from /usr/local/cuda/include/thrust/detail/config/config.h:27:0,
                     from /usr/local/cuda/include/thrust/detail/config.h:23,
                     from /usr/local/cuda/include/thrust/device_vector.h:24,
                     from /hpcfs/juno/junogpu/blyth/opticks/thrustrap/TSparse.hh:36,
                     from /hpcfs/juno/junogpu/blyth/opticks/thrustrap/TSparse_.cu:29:
    /usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:92:13: warning: Thrust requires GCC 5.0. Please upgrade your compiler. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message. [enabled by default]
          THRUST_COMPILER_DEPRECATION(GCC 5.0, upgrade your compiler);
                 ^
    /usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:104:13: warning: Thrust requires C++14. Please pass -std=c++14 to your compiler. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message. [enabled by default]
       THRUST_COMPILER_DEPRECATION(C++14, pass -std=c++14 to your compiler);
                 ^
    In file included from /usr/local/cuda/include/thrust/detail/config/config.h:27:0,
                     from /usr/local/cuda/include/thrust/detail/config.h:23,
                     from /usr/local/cuda/include/thrust/iterator/counting_iterator.h:34,
                     from /hpcfs/juno/junogpu/blyth/opticks/thrustrap/strided_range.h:52,
                     from /hpcfs/juno/junogpu/blyth/opticks/thrustrap/TBufPair_.cu:22:
    /usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:92:13: warning: Thrust requires GCC 5.0. Please upgrade your compiler. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message. [enabled by default]
          THRUST_COMPILER_DEPRECATION(GCC 5.0, upgrade your compiler);
                 ^
    /usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:104:13: warning: Thrust requires C++14. Please pass -std=c++14 to your compiler. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message. [enabled by default]
       THRUST_COMPILER_DEPRECATION(C++14, pass -std=c++14 to your compiler);
                 ^
    In file included from /usr/local/cuda/include/cub/util_arch.cuh:36:0,
                     from /usr/local/cuda/include/thrust/system/cuda/detail/util.h:32,
                     from /usr/local/cuda/include/thrust/system/cuda/detail/for_each.h:34,
                     from /usr/local/cuda/include/thrust/system/detail/adl/for_each.h:42,
                     from /usr/local/cuda/include/thrust/detail/for_each.inl:27,
                     from /usr/local/cuda/include/thrust/for_each.h:279,
                     from /hpcfs/juno/junogpu/blyth/opticks/thrustrap/TRngBuf_.cu:23:
    /usr/local/cuda/include/cub/util_cpp_dialect.cuh:117:13: warning: CUB requires GCC 5.0. Please upgrade your compiler. Define CUB_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message. [enabled by default]
          CUB_COMPILER_DEPRECATION(GCC 5.0, upgrade your compiler);
                 ^
    In file included from /usr/local/cuda/include/cub/util_arch.cuh:36:0,
                     from /usr/local/cuda/include/thrust/system/cuda/detail/util.h:32,
                     from /usr/local/cuda/include/thrust/system/cuda/detail/for_each.h:34,
                     from /usr/local/cuda/include/thrust/system/detail/adl/for_each.h:42,
                     from /usr/local/cuda/include/thrust/detail/for_each.inl:27,
                     from /usr/local/cuda/include/thrust/for_each.h:279,
                     from /hpcfs/juno/junogpu/blyth/opticks/thrustrap/TRngBuf_.cu:23:
    /usr/local/cuda/include/cub/util_cpp_dialect.cuh:129:13: warning: CUB requires C++14. Please pass -std=c++14 to your compiler. Define CUB_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message. [enabled by default]
       CUB_COMPILER_DEPRECATION(C++14, pass -std=c++14 to your compiler);
                 ^
    In file included from /usr/local/cuda/include/cub/util_arch.cuh:36:0,
                     from /usr/local/cuda/include/thrust/system/cuda/detail/util.h:32,
                     from /usr/local/cuda/include/thrust/system/cuda/detail/for_each.h:34,
                     from /usr/local/cuda/include/thrust/system/detail/adl/for_each.h:42,
                     from /usr/local/cuda/include/thrust/detail/for_each.inl:27,
                     from /usr/local/cuda/include/thrust/for_each.h:279,
                     from /usr/local/cuda/include/thrust/system/detail/generic/transform.inl:19,
                     from /usr/local/cuda/include/thrust/system/detail/generic/transform.h:105,
                     from /usr/local/cuda/include/thrust/detail/transform.inl:25,
                     from /usr/local/cuda/include/thrust/transform.h:724,
                     from /usr/local/cuda/include/thrust/system/detail/generic/copy.inl:23,
                     from /usr/local/cuda/include/thrust/system/detail/generic/copy.h:58,
                     from /usr/local/cuda/include/thrust/detail/copy.inl:21,
                     from /usr/local/cuda/include/thrust/detail/copy.h:90,
                     from /usr/local/cuda/include/thrust/detail/allocator/copy_construct_range.inl:21,
                     from /usr/local/cuda/include/thrust/detail/allocator/copy_construct_range.h:46,
                     from /usr/local/cuda/include/thrust/detail/contiguous_storage.inl:23,
                     from /usr/local/cuda/include/thrust/detail/contiguous_storage.h:241,
                     from /usr/local/cuda/include/thrust/detail/vector_base.h:30,
                     from /usr/local/cuda/include/thrust/device_vector.h:25,
                     from /hpcfs/juno/junogpu/blyth/opticks/thrustrap/TSparse.hh:36,
                     from /hpcfs/juno/junogpu/blyth/opticks/thrustrap/TSparse_.cu:29:
    /usr/local/cuda/include/cub/util_cpp_dialect.cuh:117:13: warning: CUB requires GCC 5.0. Please upgrade your compiler. Define CUB_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message. [enabled by default]
          CUB_COMPILER_DEPRECATION(GCC 5.0, upgrade your compiler);
                 ^

