cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name U4Test)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   


#[=[
Unlike most executables U4VPrimaryGeneratorTest 
needs the MOCK_CURAND macro in order to use CUDA 
curand using code on the CPU by mocking the curand
functions
#]=]

set(TGT U4VPrimaryGeneratorTest)
add_executable(${TGT} U4VPrimaryGeneratorTest.cc)
target_compile_definitions( ${TGT} PRIVATE MOCK_CURAND ) 
target_link_libraries(${TGT} U4)
install(TARGETS ${TGT} DESTINATION lib)


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



set(U4RecorderTest_SOURCES
   U4RecorderTest.cc
   DsG4Scintillation.cc
   G4Cerenkov_modified.cc
)


set(TGT U4RecorderTest)
add_executable(${TGT} ${U4RecorderTest_SOURCES})

target_link_libraries(${TGT} U4)
install(TARGETS ${TGT} DESTINATION lib)

