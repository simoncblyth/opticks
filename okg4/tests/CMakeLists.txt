cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OKG4Test)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)


set(TEST_SOURCES
)

set(_TEST_SOURCES    # create the executable by do not add_test for these
    OKX4Test.cc
    OKG4Test.cc
)

set(FORMER_TEST_SOURCES
)



foreach(SRC ${TEST_SOURCES})

    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} OKG4)
    install(TARGETS ${TGT} DESTINATION lib)

    if(${SRC} IN_LIST TEST_SOURCES)   # CMake 3.3
        add_test(${name}.${TGT} ${TGT})
    endif()

endforeach()





