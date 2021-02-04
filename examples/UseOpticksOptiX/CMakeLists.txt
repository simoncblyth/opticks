cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOpticksOptiX)
project(${name})
include(OpticksBuildOptions)


find_package(OpticksCUDA MODULE REQUIRED)

set(OpticksOptiX_VERBOSE ON)
find_package(OpticksOptiX MODULE REQUIRED)   

if(OpticksOptiX_FOUND)
    message(STATUS "OpticksOptiX_MODULE  : ${OpticksOptiX_MODULE} ")  
    message(STATUS "OpticksOptiX_FOUND   : ${OpticksOptiX_FOUND} ")
    message(STATUS "OpticksOptiX_VERSION : ${OpticksOptiX_VERSION} ")
    message(STATUS "OpticksOptiX_CONFIG  : ${OpticksOptiX_CONFIG} ")

else()
    message(STATUS "OpticksOptiX not found")
endif()



add_executable(${name} ${name}.cc)
target_link_libraries(${name} 
     PUBLIC 
         Opticks::OptiX
         Opticks::OpticksCUDA   
)
install(TARGETS ${name}  DESTINATION lib)



