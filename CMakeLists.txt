cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(Opticks)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules") 
set(OptiX_INSTALL_DIR "/Developer/OptiX" CACHE PATH "Path to OptiX installed location.")

set(SUPERBUILD ON)

include(CTest)
include(CPack)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})

include(EnvBuildOptions)

if(DUMP)
   # https://cmake.org/Wiki/CMake_Useful_Variables
   message("CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
   message("CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
   message("CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")
   message("CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")
   message("CMAKE_CXX_FLAGS_RELWITHDEBINFO= ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

find_package(OpticksBoost REQUIRED QUIET)

add_subdirectory(boostrap)    
#add_subdirectory(boostrapclient)    
add_subdirectory(numerics/npy)   
#add_subdirectory(numerics/npyclient)   

#set(REST ON)
if(REST)
   add_subdirectory(optickscore)         
   add_subdirectory(optix/ggeo)     
   add_subdirectory(graphics/assimprap)   
   add_subdirectory(graphics/openmeshrap)  
   add_subdirectory(opticksgeo)  
   add_subdirectory(graphics/oglrap)       

   find_package(CUDA ${OPTICKS_CUDA_VERSION})
   find_package(OptiX ${OPTICKS_OPTIX_VERSION})

   if(CUDA_FOUND AND OptiX_FOUND)
       set(WITH_OPTIX ON)
       add_subdirectory(cuda/cudarap)        
       add_subdirectory(numerics/thrustrap)
       add_subdirectory(graphics/optixrap)
       add_subdirectory(opticksop)
       add_subdirectory(opticksgl)
   else(CUDA_FOUND AND OptiX_FOUND)
       set(WITH_OPTIX OFF)
       message("Failed to find at least CUDA ${OPTICKS_CUDA_VERSION} and OptiX ${OPTICKS_OPTIX_VERSION} ")
       message("proceeding with viz only configure.") 
   endif()
   add_subdirectory(graphics/ggeoview)

   find_package(Geant4 QUIET)
   if(Geant4_FOUND)
      add_subdirectory(optix/cfg4)
   else(Geant4_FOUND)
      message("No Geant4 ")
   endif(Geant4_FOUND)

endif(REST)



