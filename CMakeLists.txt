cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(OPTICKS)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

add_subdirectory(boost/bpo/bcfg)
add_subdirectory(boost/bregex)
add_subdirectory(numerics/npy)
add_subdirectory(opticks)
add_subdirectory(optix/ggeo)
add_subdirectory(graphics/assimpwrap)
add_subdirectory(graphics/openmeshrap)
add_subdirectory(graphics/oglrap)

#find_package(OptiX QUIET)
#if(OptiX_FOUND)

add_subdirectory(cuda/cudawrap)
add_subdirectory(numerics/thrustrap)
add_subdirectory(graphics/optixrap)
add_subdirectory(opticksop)

#endif()

add_subdirectory(graphics/ggeoview)

#add_subdirectory(optix/cfg4)

