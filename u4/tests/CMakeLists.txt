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
   U4MaterialTest.cc
   U4MaterialPropertyVectorTest.cc
   U4PhotonInfoTest.cc
   U4GDMLTest.cc
   U4PhysicalConstantsTest.cc
   U4RandomTest.cc
   U4VolumeMakerTest.cc
   U4LogTest.cc
   U4RotationMatrixTest.cc
   U4TransformTest.cc
   U4TraverseTest.cc
   U4TreeTest.cc
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})

    target_link_libraries(${TGT} U4)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()




#[=[
Note that the U4RecorderTest executable also 
needs the MOCK_CURAND macro in order to use CUDA 
curand using code on the CPU by mocking the curand
functions
#]=]

set(U4RecorderTest_SOURCES
   U4RecorderTest.cc
   DsG4Scintillation.cc
   ShimG4OpAbsorption.cc
   ShimG4OpRayleigh.cc
   G4Cerenkov_modified.cc
   U4Physics.cc
)

set(TGT U4RecorderTest)
add_executable(${TGT} ${U4RecorderTest_SOURCES})
target_compile_definitions( ${TGT} PRIVATE MOCK_CURAND ) 
target_compile_definitions( ${TGT} PUBLIC DEBUG_TAG ) 
target_link_libraries(${TGT} U4)
install(TARGETS ${TGT} DESTINATION lib)



set(U4HitTest_SOURCES
    U4HitTest.cc
)


find_package(CSG REQUIRED)
set(TGT U4HitTest)
add_executable(${TGT} ${U4HitTest_SOURCES})
target_link_libraries(${TGT} U4 Opticks::CSG)
install(TARGETS ${TGT} DESTINATION lib)




