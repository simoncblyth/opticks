cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseYoctoGL)
project(${name})
include(OpticksBuildOptions)

find_package(YoctoGL REQUIRED CONFIG)  # from /usr/local/opticks/externals/lib/cmake/yoctogl/  

set(YoctoGL_VERBOSE OFF)
if(YoctoGL_VERBOSE)
    message(STATUS "YoctoGL_DIR   : ${YoctoGL_DIR} ")
    message(STATUS "YoctoGL_FOUND : ${YoctoGL_FOUND} ")
    echo_target_std(Opticks::YoctoGL)
endif()

set(TEST_SOURCES
   ${name}.cc
   ${name}_Write.cc
   ${name}_json.cc
)

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TGT ${TEST_SOURCE} NAME_WE)
    add_executable(${TGT} ${TEST_SOURCE})
    target_link_libraries(${TGT} PUBLIC Opticks::YoctoGL)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()



