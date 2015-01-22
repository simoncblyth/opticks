cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(name OptiXTest)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{OPTIX_SDK_DIR}/CMake"
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

find_package(OptiX REQUIRED)

include(FindSUtilGLUT)
include(EnvBuildOptions)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OptiX_INCLUDE}  
    $ENV{OPTIX_SDK_DIR}/sutil
)


add_definitions(
    ${OptiX_DEFINITIONS} 
)


add_executable(${name} ${name}.cc )

target_link_libraries( ${name} 
                        /usr/local/env/cuda/OptiX_301/lib/libsutil.dylib
                        optix
                      )

install(TARGETS ${name} DESTINATION bin)
