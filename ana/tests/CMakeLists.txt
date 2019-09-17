cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name AnaTests)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)


find_package(SysRap REQUIRED CONFIG)

set(TEST_SOURCES
    AnaTest.cc
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})

    target_link_libraries(${TGT} PUBLIC 
          Opticks::SysRap
    )
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()




