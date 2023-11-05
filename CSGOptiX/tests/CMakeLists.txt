cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CSGOptiXTest)
project(${name} VERSION 0.1.0)

#[=[
Promote sources to ctests by moving to lower list 
#]=]

set(SINGLE_SOURCES
    CSGOptiXSimtraceTest.cc
    CSGOptiXSimulateTest.cc
    CSGOptiXSimTest.cc
    CXRaindropTest.cc

    CSGOptiXRMTest.cc
    CSGOptiXTMTest.cc
    CSGOptiXSMTest.cc
)

set(TEST_SOURCES
    CSGOptiXVersion.cc
    CSGOptiXVersionTest.cc
    CSGOptiXRenderTest.cc
)

foreach(SRC ${SINGLE_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} CSGOptiX)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    add_test(${name}.${TGT} ${TGT})
    target_link_libraries(${TGT} CSGOptiX)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


