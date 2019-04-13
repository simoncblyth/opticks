cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseInstanceTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

set(TESTS
    UseInstanceTest.cc
    OneTriangleTest.cc
)

foreach(SRC ${TESTS})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})
  
    target_link_libraries(${TGT} UseInstance)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()

