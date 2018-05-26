
# start from nothing, so repeated inclusion of this into CMake context doesnt repeat the flags 
set(CMAKE_CXX_FLAGS)
set(CUDA_NVCC_FLAGS)

if(WIN32)

  # need to detect compiler not os?
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W4") # overall warning level 4
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd4996")   # disable  C4996: 'strdup': The POSIX name for this item is deprecated.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_CRT_SECURE_NO_WARNINGS")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ITERATOR_DEBUG_LEVEL=0")


else(WIN32)

  ## c++11 forced by AsioZMQ : AsioZMQ not used here, but expect best to use same compiler options as far as possible
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++0x")  ## huh nvcc compilation fails with this ???
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
     # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -stdlib=libc++")
     set(CMAKE_CXX_STANDARD 14)
     set(CMAKE_CXX_STANDARD_REQUIRED on)
  else ()
      #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
      #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x")
     # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")   #needed for numpyserver- on Linux ?
     set(CMAKE_CXX_STANDARD 14)
     set(CMAKE_CXX_STANDARD_REQUIRED on)

  endif ()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")

  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-comment")
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shadow")
  else()
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shadow")
  endif()

endif(WIN32)



set(COMPUTE_CAPABILITY 30)
if(NOT (COMPUTE_CAPABILITY LESS 30))


   #list(APPEND CUDA_NVCC_FLAGS "-arch=sm_${COMPUTE_CAPABILITY}")
   list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
   list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}")

   #list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
   # https://github.com/facebookresearch/Detectron/issues/185


   list(APPEND CUDA_NVCC_FLAGS "-O2")
   #list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE")
   list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
   # see env-/nvcc- for background on flags  

   #list(APPEND CUDA_NVCC_FLAGS "-m64")
   #list(APPEND CUDA_NVCC_FLAGS "--disable-warnings")

   # CUDA headers yield many:
   # /usr/local/cuda/include/device_functions.hpp:283:3:   warning: extension used [-Wlanguage-extension-token]
   # TODO: find way to selectively disable warnings

   # https://cmake.org/cmake/help/v3.0/module/FindCUDA.html
   set(CUDA_PROPAGATE_HOST_FLAGS OFF)
   set(CUDA_VERBOSE_BUILD OFF)

endif()
 

if(FLAGS_VERBOSE)
   # https://cmake.org/Wiki/CMake_Useful_Variables
   message(STATUS "OpticksCompilationFlags.cmake : COMPUTE_CAPABILITY : ${COMPUTE_CAPABILITY}")
   message(STATUS "OpticksCompilationFlags.cmake : CUDA_NVCC_FLAGS    : ${CUDA_NVCC_FLAGS} ")
   message(STATUS "OpticksCompilationFlags.cmake : CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
   message(STATUS "OpticksCompilationFlags.cmake : CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
   message(STATUS "OpticksCompilationFlags.cmake : CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")
   message(STATUS "OpticksCompilationFlags.cmake : CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")
   message(STATUS "OpticksCompilationFlags.cmake : CMAKE_CXX_FLAGS_RELWITHDEBINFO= ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
   message(STATUS "OpticksCompilationFlags.cmake : CMAKE_CXX_STANDARD : ${CMAKE_CXX_STANDARD} " )
   message(STATUS "OpticksCompilationFlags.cmake : CMAKE_CXX_STANDARD_REQUIRED : ${CMAKE_CXX_STANDARD_REQUIRED} " )
endif()



