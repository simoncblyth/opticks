cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OKConfTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)


set(TEST_SOURCES
    OKConfTest.cc
)

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TGT ${TEST_SOURCE} NAME_WE)
    add_executable(${TGT} ${TEST_SOURCE})

    set(testname ${name}.${TGT})  

    # see cmakex-
    add_test(${testname} ${TGT})

    target_link_libraries(${TGT} Opticks::OKConf) 
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()



