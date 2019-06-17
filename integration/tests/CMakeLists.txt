cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name IntegrationTests)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)


find_package(SysRap REQUIRED CONFIG)

set(TEST_SOURCES
    IntegrationTest.cc
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



set(SCRIPT tboolean.sh)
get_filename_component(TGT ${SCRIPT} NAME_WE)
set(ARG box)

set(TEST ${name}.${TGT}.${ARG})
add_test(${TEST} ${SCRIPT} ${ARG} )
#set_tests_properties(${TEST} PROPERTIES ENVIRONMENT "SEnvTest_COLOR=${COLOR}")


#[=[
Debug fabricated tests with::

  ctest -R IntegrationTests.tboolean. --output-on-failure 


#]=]

