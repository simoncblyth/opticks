cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseYoctoGLRap)
project(${name})
include(OpticksBuildOptions)

find_package(YoctoGLRap REQUIRED CONFIG)  # from /usr/local/opticks/externals/lib/cmake/yoctoglrap/  

set(TEST_SOURCES
   ${name}.cc
)

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TGT ${TEST_SOURCE} NAME_WE)
    add_executable(${TGT} ${TEST_SOURCE})
    target_link_libraries(${TGT} PUBLIC Opticks::YoctoGLRap)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()



