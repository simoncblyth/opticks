cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name G4CXTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

set(INSTALLED_SCRIPTS
    GXTestRunner.sh 
)
install(PROGRAMS ${INSTALLED_SCRIPTS} DESTINATION bin)

#[=[
Promote to ctests by moving between the lists 
Issues preventing them being ctests are:

1. need for environment setup 
2. dependency on geometry specifics

So generally the less config the easier to become tests. 
#]=]


set(SINGLE_SOURCES

   G4CXSimtraceTest.cc

   G4CXSimulateTest.cc
   G4CXTest.cc 

   G4CX_U4TreeCreateCSGFoundryTest.cc 

   G4CXOpticks_SetGeometry_GetInputPhoton_Test.cc
)

set(TEST_SOURCES
   G4CXRenderTest.cc
   G4CXOpticks_setGeometry_Test.cc
)


foreach(SRC ${SINGLE_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} G4CX)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


#find_program(BASH_EXECUTABLE NAMES bash REQUIRED)
#message(STATUS "BASH_EXECUTABLE : ${BASH_EXECUTABLE}")

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} G4CX)
    install(TARGETS ${TGT} DESTINATION lib)

    #add_test(${name}.${TGT} ${TGT})
    add_test(
       NAME ${name}.${TGT} 
       COMMAND bash GXTestRunner.sh ${TGT}
       #COMMAND ${BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/GXTestRunner.sh ${TGT}
    )
endforeach()







