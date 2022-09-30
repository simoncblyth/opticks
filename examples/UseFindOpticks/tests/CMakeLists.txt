cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseFindOpticksTest)
project(${name} VERSION 0.1.0)


set(TEST_SOURCES
   DemoLibTest.cc
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} UseFindOpticks)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()

