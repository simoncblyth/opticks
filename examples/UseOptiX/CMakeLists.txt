cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOptiX)
project(${name})
include(OpticksBuildOptions)

#set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#[=[
opticks-llp '$ORIGIN/..'

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/OptiX/lib64")
 
This setting is now done as standard by OpticksBuildOptions.cmake

#]=]


set(OptiX_VERBOSE ON)
find_package(OptiX MODULE QUIET)   
find_package(OpticksCUDA  MODULE QUIET)   

if(OptiX_FOUND)
    message(STATUS "OptiX_MODULE  : ${OptiX_MODULE} ")  
    message(STATUS "OptiX_FOUND   : ${OptiX_FOUND} ")
    message(STATUS "OptiX_VERSION : ${OptiX_VERSION} ")
    message(STATUS "OptiX_CONFIG  : ${OptiX_CONFIG} ")

    echo_pfx_vars(OptiX "FOUND;VERSION;CONFIG;INCLUDE;LIBRARY;DLL;INSTALL_DIR")

    #echo_target_std(Opticks::OptiX)
    #echo_target_std(optix)
    #echo_target_std(optixu)
    #echo_target_std(optix_prime)
    #echo_pfx_vars(optix "rpath;LIBRARY")

else()
    message(STATUS "OptiX not found")
endif()

add_executable(${name} ${name}.cc)
target_link_libraries(${name} 
     PUBLIC 
         Opticks::OptiX
         Opticks::OpticksCUDA
)
install(TARGETS ${name}  DESTINATION lib)

# to take this further, see /Developer/OptiX/SDK/CMakeLists.txt:OPTIX_add_sample_executable

