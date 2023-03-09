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
   Deprecated_U4PhotonInfoTest.cc
   U4TrackInfoTest.cc 
   U4TrackTest.cc
   U4GDMLTest.cc
   U4GDMLReadTest.cc 
   U4PhysicalConstantsTest.cc
   U4RandomTest.cc
   U4UniformRandTest.cc
   U4EngineTest.cc
   U4VolumeMakerTest.cc
   U4LogTest.cc
   U4RotationMatrixTest.cc
   U4TransformTest.cc
   U4TraverseTest.cc
   U4TreeTest.cc
   U4TreeCreateTest.cc
   U4Material_MakePropertyFold_MakeTest.cc
   U4Material_MakePropertyFold_LoadTest.cc
   U4TouchableTest.cc
   U4SurfaceTest.cc
   U4SolidTest.cc 

   U4Debug_Test.cc 
   U4Hit_Debug_Test.cc 

   G4ThreeVectorTest.cc 
 
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



message(STATUS "${CMAKE_CURRENT_LIST_FILE} : PMTFastSim_FOUND ${PMTFastSim_FOUND}")


## HMM: its kinda confusing to not build an executable 
## depending on the switch because when flipping the switch
## off the old executable still runs, but ignores and changes made 
## its better to always build the executable and have it 
## fail with an informative when needed switches are not set 

set(PMTFastSim_TESTS
   U4SimulateTest.cc
   U4SimtraceTest.cc
   U4PMTFastSimTest.cc
   U4PMTAccessorTest.cc
)

foreach(SRC ${PMTFastSim_TESTS})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_compile_definitions( ${TGT} PRIVATE MOCK_CURAND ) 
    target_compile_definitions( ${TGT} PRIVATE WITH_NP )    ## for _PMTSimParamData 
    target_compile_definitions( ${TGT} PRIVATE POM_DEBUG )   

    target_link_libraries(${TGT} U4 Opticks::SysRap)

    if(PMTFastSim_FOUND)
       target_link_libraries(${TGT} Opticks::PMTFastSim)
    endif()

    install(TARGETS ${TGT} DESTINATION lib)
endforeach()





