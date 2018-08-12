#[=[

See env-/nvcc- for background on flags  

#]=]

set(CUDA_NVCC_FLAGS)

if(NOT (COMPUTE_CAPABILITY LESS 30))

   #list(APPEND CUDA_NVCC_FLAGS "-arch=sm_${COMPUTE_CAPABILITY}")
   list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
   list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}")

   #list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
   # https://github.com/facebookresearch/Detectron/issues/185

   list(APPEND CUDA_NVCC_FLAGS "-O2")
   #list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE")
   list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

   #list(APPEND CUDA_NVCC_FLAGS "-m64")
   #list(APPEND CUDA_NVCC_FLAGS "--disable-warnings")

   set(CUDA_PROPAGATE_HOST_FLAGS OFF)
   set(CUDA_VERBOSE_BUILD OFF)

endif()
 

if(FLAGS_VERBOSE)
   message(STATUS "OpticksCUDAFlags.cmake : COMPUTE_CAPABILITY : ${COMPUTE_CAPABILITY}")
   message(STATUS "OpticksCUDAFlags.cmake : CUDA_NVCC_FLAGS    : ${CUDA_NVCC_FLAGS} ")
endif()



