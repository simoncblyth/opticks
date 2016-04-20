cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(OPTICKS)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

add_subdirectory(boost/bpo/bcfg)  
add_subdirectory(boost/bregex)    
add_subdirectory(numerics/npy)   

add_subdirectory(opticks)         # Opticks : Boost GLM NPY Cfg Bregex GGeo 
add_subdirectory(optix/ggeo)      # GGeo    : Boost Bregex GLM NPY Opticks

# unhealthy GGeo/Opticks cross dependency... 
#  TODO: move PICKFACE handling from CompositionCfg higher up ? maybe to GGeoCfg ?
#        so can remove GGeo dep from Opticks
#

add_subdirectory(graphics/assimpwrap)   # AssimpWrap
add_subdirectory(graphics/openmeshrap)  # OpenMeshRap
add_subdirectory(graphics/oglrap)       # OGLRap 

#find_package(OptiX QUIET)
#if(OptiX_FOUND)

add_subdirectory(cuda/cudawrap)        
add_subdirectory(numerics/thrustrap)
add_subdirectory(graphics/optixrap)
add_subdirectory(opticksop)
add_subdirectory(opticksgl)

#endif()

add_subdirectory(graphics/ggeoview)

#add_subdirectory(optix/cfg4)

