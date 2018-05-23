cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOptiXRap)
project(${name} VERSION 0.1.0)
    
include(EchoTarget)
set(OptiX_VERBOSE ON)

find_package(OptiXRap REQUIRED CONFIG)   

if(OptiXRap_FOUND)
    message(STATUS "OptiXRap_DIR     : ${OptiXRap_DIR} ")  
    message(STATUS "OptiXRap_VERSION : ${OptiXRap_VERSION} ")
    message(STATUS "OptiXRap_CONFIG  : ${OptiXRap_CONFIG} ")

    echo_pfx_vars(OptiXRap "FOUND;VERSION;CONFIG;INCLUDE;LIBRARY;DLL;INSTALL_DIR")
    echo_target_std(Opticks::OptiXRap)

else()
    message(STATUS "OptiX not found")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::OptiXRap )
install(TARGETS ${name}  DESTINATION lib)

# to take this further, see /Developer/OptiX/SDK/CMakeLists.txt:OPTIX_add_sample_executable

