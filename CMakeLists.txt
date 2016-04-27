cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(Opticks)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules") 
set(OptiX_INSTALL_DIR "/Developer/OptiX" CACHE PATH "Path to OptiX installed location.")

set(OPTICKS_CUDA_VERSION 5.5)
set(OPTICKS_OPTIX_VERSION 3.5)

set(SUPERBUILD ON)

include(CTest)
include(CPack)

include(EnvBuildOptions)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})

add_subdirectory(boost/bpo/bcfg)  
add_subdirectory(boost/bregex)    
add_subdirectory(numerics/npy)   

add_subdirectory(optickscore)         
add_subdirectory(optix/ggeo)     
add_subdirectory(graphics/assimprap)   
add_subdirectory(graphics/openmeshrap)  
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
    message("Failed to find at least CUDA ${OPTICKS_CUDA_VERSION} and OptiX ${OPTICKS_OPTIX_VERSION} : proceeding with viz only configure.") 
endif()

add_subdirectory(graphics/ggeoview)

#add_subdirectory(optix/cfg4)





