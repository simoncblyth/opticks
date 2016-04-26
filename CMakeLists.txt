cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(Opticks)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules") 

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

if(WITH_OPTIX)
    add_subdirectory(cuda/cudarap)        
    add_subdirectory(numerics/thrustrap)
    add_subdirectory(graphics/optixrap)
    add_subdirectory(opticksop)
    add_subdirectory(opticksgl)
endif()

add_subdirectory(graphics/ggeoview)

#add_subdirectory(optix/cfg4)





