cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name Y4CSGTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(TESTS
    Y4CSGTest.cc
)

foreach(SRC ${TESTS})

    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} Y4CSG)
    install(TARGETS ${TGT} DESTINATION lib)
    add_test(${name}.${TGT} ${TGT})

endforeach()





