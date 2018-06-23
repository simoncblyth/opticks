cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CFG4Test)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   


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
    CVecTest.cc

    G4MaterialPropertiesTableTest.cc

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

# generate targets for these, but dont add_test 
set(_TEST_SOURCES

    CX4GDMLTest.cc
)


foreach(SRC ${TEST_SOURCES} ${_TEST_SOURCES})

    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} CFG4)
    install(TARGETS ${TGT} DESTINATION lib)

    if(${SRC} IN_LIST TEST_SOURCES)   # CMake 3.3
       add_test(${name}.${TGT} ${TGT})
    endif()
endforeach()


