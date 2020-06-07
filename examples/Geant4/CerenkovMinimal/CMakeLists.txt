cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CerenkovMinimal)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()


find_package( G4OK CONFIG REQUIRED )

set(SOURCES
    DetectorConstruction.cc
    PhysicsList.cc
    L4Cerenkov.cc
    PrimaryGeneratorAction.cc
    SteppingAction.cc
    TrackingAction.cc
    EventAction.cc
    RunAction.cc
    Ctx.cc
    G4.cc
    SensitiveDetector.cc
    OpHit.cc
)

set(HEADERS
    DetectorConstruction.hh
    PhysicsList.hh
    L4Cerenkov.hh
    PrimaryGeneratorAction.hh
    SteppingAction.hh
    TrackingAction.hh
    EventAction.hh
    RunAction.hh
    Ctx.hh
    G4.hh
    SensitiveDetector.hh
    OpHit.hh
)

add_executable(${name} ${name}.cc ${SOURCES} ${HEADERS})

target_link_libraries(${name} Opticks::G4OK )
target_compile_definitions( ${name} 
  PRIVATE 
      WITH_OPTICKS
      G4USE_STD11 
 )

install(TARGETS ${name} DESTINATION lib)



