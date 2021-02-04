cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOpticksOptiX)
project(${name})
include(OpticksBuildOptions)


set(OpticksOptiX_VERBOSE ON)
find_package(OpticksOptiX MODULE QUIET)   

if(OpticksOptiX_FOUND)
    message(STATUS "OpticksOptiX_MODULE  : ${OpticksOptiX_MODULE} ")  
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
    message(STATUS "OpticksOptiX not found")
endif()


#[=[

add_executable(${name} ${name}.cc)
target_link_libraries(${name} 
     PUBLIC 
         Opticks::OptiX
         Opticks::OpticksCUDA
)
install(TARGETS ${name}  DESTINATION lib)

#]=]


