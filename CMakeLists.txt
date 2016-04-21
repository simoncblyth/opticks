cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(OPTICKS)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
                      "$ENV{OPTIX_SDK_DIR}/CMake"
          ) 

include(EnvBuildOptions)

add_subdirectory(boost/bpo/bcfg)  
add_subdirectory(boost/bregex)    
add_subdirectory(numerics/npy)   
add_subdirectory(opticks)         
add_subdirectory(optix/ggeo)     
add_subdirectory(graphics/assimpwrap)   
add_subdirectory(graphics/openmeshrap)  
add_subdirectory(graphics/oglrap)       

if(EXISTS $ENV{OPTIX_SDK_DIR}/CMake AND WITH_OPTIX)
    message("OPTIX_SDK_DIR:$ENV{OPTIX_SDK_DIR}")
    add_subdirectory(cuda/cudawrap)        
    add_subdirectory(numerics/thrustrap)
    add_subdirectory(graphics/optixrap)
    add_subdirectory(opticksop)
    add_subdirectory(opticksgl)
else()
    message("WITH_OPTIX:${WITH_OPTIX} OPTIX_SDK_DIR:$ENV{OPTIX_SDK_DIR} ")
endif()

add_subdirectory(graphics/ggeoview)

#add_subdirectory(optix/cfg4)

