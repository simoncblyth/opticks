cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OpticksGLTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

set(TEST_SOURCES
    OOAxisAppCheck.cc
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    #if (${TGT} MATCHES ".*Test$")  
       add_test(${name}.${TGT} ${TGT})
       #add_dependencies(check ${TGT})
    #endif()

    target_link_libraries(${TGT} OpticksGL )
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()

