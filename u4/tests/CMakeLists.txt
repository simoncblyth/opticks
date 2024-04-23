cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name U4Test)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

set(INSTALLED_SCRIPTS
    U4TestRunner.sh 
)
install(PROGRAMS ${INSTALLED_SCRIPTS} DESTINATION bin)


#[=[
Unlike most executables U4VPrimaryGeneratorTest 
needs the MOCK_CURAND macro in order to use CUDA 
curand using code on the CPU by mocking the curand
functions

HMM: its kinda confusing to not build an executable 
depending on switches because when flipping the switch OFF
the old executable still runs, but it is not updated and 
so it ignores any code changes made.

It is better to always build the executable and have it 
fail with an informative message when any needed switches 
are not defined.  

#]=]

set(TGT U4VPrimaryGeneratorTest)
add_executable(${TGT} U4VPrimaryGeneratorTest.cc)
target_compile_definitions( ${TGT} PRIVATE MOCK_CURAND ) 
target_link_libraries(${TGT} U4)
install(TARGETS ${TGT} DESTINATION lib)

set(TEST_SOURCES
   U4Custom4Test.cc
   U4NistManagerTest.cc
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
   U4RandomMonitorTest.cc
   U4RandomArrayTest.cc


   U4VolumeMakerTest.cc
   U4LogTest.cc
   U4RotationMatrixTest.cc
   U4TransformTest.cc
   U4TraverseTest.cc


   U4Material_MakePropertyFold_MakeTest.cc
   U4Material_MakePropertyFold_LoadTest.cc
   U4TouchableTest.cc
   U4SurfaceTest.cc
   U4SolidTest.cc 

   U4SensitiveDetectorTest.cc 

   U4Debug_Test.cc 
   U4Hit_Debug_Test.cc 

   G4ThreeVectorTest.cc 


   U4PhysicsTableTest.cc 
)

set(EXPECTED_TO_FAIL_TEST_SOURCES
   U4NavigatorTest.cc
)

set(SLOW_TEST_SOURCES
   U4TreeTest.cc
   U4TreeCreateTest.cc
   U4TreeCreateSSimTest.cc
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} U4)
    install(TARGETS ${TGT} DESTINATION lib)

    add_test(
       NAME ${name}.${TGT} 
       COMMAND bash U4TestRunner.sh ${TGT}
    )
endforeach()

foreach(SRC ${EXPECTED_TO_FAIL_TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} U4)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()

foreach(SRC ${SLOW_TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} U4)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()






#[=[
Note that the U4AppTest executable also 
needs the MOCK_CURAND macro in order to use CUDA 
curand using code on the CPU by mocking the curand
functions
#]=]

set(U4AppTest_SOURCES
   U4AppTest.cc
)

set(TGT U4AppTest)
add_executable(${TGT} ${U4AppTest_SOURCES})
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

message(STATUS "${CMAKE_CURRENT_LIST_FILE} : PMTSim_FOUND     : ${PMTSim_FOUND}")
message(STATUS "${CMAKE_CURRENT_LIST_FILE} : Custom4_FOUND    : ${Custom4_FOUND}")


set(PMTSim_TESTS
   U4SimulateTest.cc
   U4SimtraceTest.cc
   U4PMTFastSimTest.cc
   U4PMTAccessorTest.cc
)

foreach(SRC ${PMTSim_TESTS})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_compile_definitions( ${TGT} PRIVATE MOCK_CURAND ) 
    target_compile_definitions( ${TGT} PRIVATE WITH_NP )    ## for _PMTSimParamData 
    target_compile_definitions( ${TGT} PRIVATE POM_DEBUG )   

    target_link_libraries(${TGT} U4 Opticks::SysRap)

    if(PMTSim_FOUND)
       target_compile_definitions( ${TGT} PRIVATE WITH_PMTSIM )   
       target_compile_definitions( ${TGT} PRIVATE PMTSIM_STANDALONE )   
       target_link_libraries(${TGT} Opticks::PMTSim)
    endif()

    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


