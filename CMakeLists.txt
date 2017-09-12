cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
   message(FATAL_ERROR "in-source build detected : DONT DO THAT")
endif()

set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX ${CMAKE_SOURCE_DIR}/cmake/Modules/Geant4MakeRules_cxx.cmake)

set(name Opticks)
project(${name})

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules") 
set(OptiX_INSTALL_DIR "/Developer/OptiX" CACHE PATH   "Path to OptiX installed location.")
set(COMPUTE_CAPABILITY "0"               CACHE STRING "GPU Compute Capability eg one of 0,30,50,52 " )

set(SUPERBUILD ON)

include(CTest)
include(CPack)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})

include(EnvBuildOptions)



set(DUMP ON)
if(DUMP)
   # https://cmake.org/Wiki/CMake_Useful_Variables
   message("CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
   message("CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
   message("CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")
   message("CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")
   message("CMAKE_CXX_FLAGS_RELWITHDEBINFO= ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

find_package(OpticksBoost REQUIRED QUIET)

add_subdirectory(sysrap)    
add_subdirectory(boostrap)    
add_subdirectory(opticksnpy)   
add_subdirectory(optickscore)         
add_subdirectory(ggeo)     
add_subdirectory(assimprap)   
add_subdirectory(openmeshrap)  
add_subdirectory(opticksgeo)  
add_subdirectory(oglrap)       


if(NOT (COMPUTE_CAPABILITY LESS 30))

   #list(APPEND CUDA_NVCC_FLAGS "-arch=sm_${COMPUTE_CAPABILITY}")
   list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
   list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}")

   list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
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

   message(STATUS "${name}.COMPUTE_CAPABILITY : ${COMPUTE_CAPABILITY}")
   message(STATUS "${name}.CUDA_NVCC_FLAGS    : ${CUDA_NVCC_FLAGS} ")

   find_package(CUDA ${OPTICKS_CUDA_VERSION})
   find_package(OptiX ${OPTICKS_OPTIX_VERSION})


   if(CUDA_FOUND AND OptiX_FOUND)
       set(WITH_OPTIX ON)
       add_subdirectory(cudarap)        
       add_subdirectory(thrustrap)
       add_subdirectory(optixrap)
       add_subdirectory(okop)
       add_subdirectory(opticksgl)
       
       message(STATUS "${name}.OXRAP_OPTIX_VERSION : ${OXRAP_OPTIX_VERSION} ")

   else(CUDA_FOUND AND OptiX_FOUND)
       set(WITH_OPTIX OFF)
       message("${name}.Failed to find at least CUDA ${OPTICKS_CUDA_VERSION} and OptiX ${OPTICKS_OPTIX_VERSION} ")
       message("${name}.proceeding with viz only configure.") 
   endif()

else(NOT (COMPUTE_CAPABILITY LESS 30))
   message(STATUS "${name}.COMPUTE_CAPABILITY : ${COMPUTE_CAPABILITY} : at least 30 is required for Opticks, proceeding GPU-less")
endif()


set(REST ON CACHE BOOL "Build OK and CFG4 packages")
if(REST)
   add_subdirectory(ok)

   find_package(Geant4 QUIET)
   if(Geant4_FOUND)
      add_subdirectory(cfg4)
      add_subdirectory(okg4)
   else(Geant4_FOUND)
      message("${name}.Geant4_FOUND_NOT")
   endif(Geant4_FOUND)

endif(REST)



# collects version defines, currently only OXRAP_OPTIX_VERSION and CFG4_G4VERSION_NUMBER
configure_file(OpticksCMakeConfig.hh.in inc/OpticksCMakeConfig.hh)

# generate opticks-config into lib dir
include(OpticksConfigureConfigScript)




