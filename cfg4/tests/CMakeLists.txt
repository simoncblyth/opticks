cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CFG4Test)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

set(_TEST_SOURCES

)

set(TEST_SOURCES

    CMaterialLibTest.cc
    CMaterialTest.cc
    CTestDetectorTest.cc
    CGDMLDetectorTest.cc
    CGeometryTest.cc
    CG4Test.cc
    G4MaterialTest.cc
    G4StringTest.cc
    G4SphereTest.cc
    G4PhysicsOrderedFreeVectorTest.cc

    G4UniformRandTest.cc
    G4BoxTest.cc
    G4ThreeVectorTest.cc
    CCollectorTest.cc
    CInterpolationTest.cc
    OpRayleighTest.cc 
    CGROUPVELTest.cc 

    CMakerTest.cc
    CPhotonTest.cc
    CRandomEngineTest.cc

)


foreach(SRC ${TEST_SOURCES})

    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} CFG4)
    install(TARGETS ${TGT} DESTINATION lib)

    add_test(${name}.${TGT} ${TGT})
    #add_dependencies(check ${TGT})

endforeach()

