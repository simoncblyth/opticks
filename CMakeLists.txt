cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
   message(FATAL_ERROR "in-source build detected : DONT DO THAT")
endif()

set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX ${CMAKE_SOURCE_DIR}/cmake/Modules/Geant4MakeRules_cxx.cmake)

set(name Opticks)
project(${name})

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules") 
set(OptiX_INSTALL_DIR "/Developer/OptiX" CACHE PATH "Path to OptiX installed location.")

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

if(APPLE)
    set(GPU ON)
endif(APPLE)
if(GPU)
   find_package(CUDA ${OPTICKS_CUDA_VERSION})
   find_package(OptiX ${OPTICKS_OPTIX_VERSION})
   if(CUDA_FOUND AND OptiX_FOUND)
       set(WITH_OPTIX ON)
       add_subdirectory(cudarap)        
       add_subdirectory(numerics/thrustrap)
       add_subdirectory(graphics/optixrap)
       add_subdirectory(opticksop)
       add_subdirectory(opticksgl)
   else(CUDA_FOUND AND OptiX_FOUND)
       set(WITH_OPTIX OFF)
       message("${name}.Failed to find at least CUDA ${OPTICKS_CUDA_VERSION} and OptiX ${OPTICKS_OPTIX_VERSION} ")
       message("${name}.proceeding with viz only configure.") 
   endif()
endif(GPU)


set(REST ON)
if(REST)
   add_subdirectory(graphics/ggeoview)

   find_package(Geant4 QUIET)
   if(Geant4_FOUND)
      add_subdirectory(optix/cfg4)
   else(Geant4_FOUND)
      message("${name}.Geant4_FOUND_NOT")
   endif(Geant4_FOUND)

endif(REST)



