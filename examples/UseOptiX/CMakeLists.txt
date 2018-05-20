cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOptiX)
project(${name})
    
include(EchoTarget)

find_package(OptiX MODULE QUIET)   

if(OptiX_FOUND)
    message(STATUS "OptiX_FOUND   : ${OptiX_FOUND} ")
    message(STATUS "OptiX_VERSION : ${OptiX_VERSION} ")

    #echo_target_std(Opticks::OptiX)
    #echo_target_std(optix)
    #echo_target_std(optixu)
    #echo_target_std(optix_prime)
    #echo_pfx_vars(optix "rpath;LIBRARY")
    #echo_pfx_vars(OptiX "INCLUDE;LIBRARY;DLL;INSTALL_DIR")

else()
    message(STATUS "OptiX not found")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::OptiX)
install(TARGETS ${name}  DESTINATION lib)

# to take this further, see /Developer/OptiX/SDK/CMakeLists.txt:OPTIX_add_sample_executable

