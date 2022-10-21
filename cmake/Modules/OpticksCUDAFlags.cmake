#[=[

See env-/nvcc- for background on flags  


https://developer.nvidia.com/cuda-toolkit-archive

https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/


CUDA Toolkit 11.8.0 (October 2022)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-altering-compiler-linker-behavior-std

4.2.3.16. --std {c++03|c++11|c++14|c++17} (-std)

Select a particular C++ dialect.
Allowed Values

    c++03
    c++11
    c++14
    c++17

Default

The default C++ dialect depends on the host compiler. nvcc matches the default C++ dialect that the host compiler uses.


CUDA Toolkit 10.1  (Using this on Linux, together with OptiX 7.0.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://docs.nvidia.com/cuda/archive/10.1/
* https://docs.nvidia.com/cuda/archive/10.1/cuda-compiler-driver-nvcc/index.html

4.2.3.11. --std {c++03|c++11|c++14} (-std)

Select a particular C++ dialect.
Allowed Values

    c++03
    c++11
    c++14

Default

The default C++ dialect depends on the host compiler. nvcc matches the default C++ dialect that the host compiler uses.


CUDA Toolkit 9.1 (Using this on macOS, together with OptiX 5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://docs.nvidia.com/cuda/archive/9.1/cuda-compiler-driver-nvcc/index.html

--std {c++03|c++11|c++14} 

Select a particular C++ dialect.

Allowed values for this option: c++03,c++11,c++14









#]=]


set(OPTICKS_CUDA_NVCC_DIALECT $ENV{OPTICKS_CUDA_NVCC_DIALECT})
if(OPTICKS_CUDA_NVCC_DIALECT)
    message(STATUS "cmake/Modules/OpticksCUDAFlags.cmake : reading envvar OPTICKS_CUDA_NVCC_DIALECT into variable ${OPTICKS_CUDA_NVCC_DIALECT}")
else()
    set(OPTICKS_CUDA_NVCC_DIALECT "c++11")
    message(STATUS "cmake/Modules/OpticksCUDAFlags.cmake : using default OPTICKS_CUDA_NVCC_DIALECT variable ${OPTICKS_CUDA_NVCC_DIALECT}")
endif()


set(CUDA_NVCC_FLAGS)

if(NOT (COMPUTE_CAPABILITY LESS 30))

   #list(APPEND CUDA_NVCC_FLAGS "-arch=sm_${COMPUTE_CAPABILITY}")
   list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
   list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}")

   list(APPEND CUDA_NVCC_FLAGS "-std=${OPTICKS_CUDA_NVCC_DIALECT}")
   # https://github.com/facebookresearch/Detectron/issues/185
   # notes/issues/g4_1062_opticks_with_newer_gcc_for_G4OpticksTest.rst 

   list(APPEND CUDA_NVCC_FLAGS "-O2")
   #list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE")
   list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

   list(APPEND CUDA_NVCC_FLAGS "-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored ") 
   # notes/issues/glm_anno_warnings_with_gcc_831.rst 

   #list(APPEND CUDA_NVCC_FLAGS "-m64")
   #list(APPEND CUDA_NVCC_FLAGS "--disable-warnings")

   set(CUDA_PROPAGATE_HOST_FLAGS OFF)
   set(CUDA_VERBOSE_BUILD OFF)

endif()
 

if(FLAGS_VERBOSE)
   message(STATUS "OpticksCUDAFlags.cmake : COMPUTE_CAPABILITY : ${COMPUTE_CAPABILITY}")
   message(STATUS "OpticksCUDAFlags.cmake : CUDA_NVCC_FLAGS    : ${CUDA_NVCC_FLAGS} ")
endif()



