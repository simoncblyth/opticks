cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CerenkovMinimal)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(OKConf_VERBOSE ON)
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
 )

install(TARGETS ${name} DESTINATION lib)



