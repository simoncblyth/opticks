cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name U4Test)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

set(TEST_SOURCES
   U4Test.cc
   U4PhotonInfoTest.cc
   U4GDMLTest.cc
)


foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})

    target_link_libraries(${TGT} U4)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()



set(TGT U4RecorderTest)
add_executable(${TGT} ${TGT}.cc DsG4Scintillation.cc)

target_link_libraries(${TGT} U4)
install(TARGETS ${TGT} DESTINATION lib)

