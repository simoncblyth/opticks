cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name ExtG4)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

#[=[
ExtG4 : X4 : Direct/Live conversion of Geant4 geometry into GGeo
=========================================================================

Class requirements for X4 package:

1. simple single objective classes that wrap single G4 instances
2. named after the classes they enhance
3. expose functionaliy via static functions where possible
4. **NOT generally useful code : focus just on doing conversion**

* **NB does not depend on the CFG4 (big gorilla) the idea behind 
  this is to force X4 to be kept minimal and stable**  

* X4 is aiming to become production code, not instrumented debugging code 
  like much of CFG4

Some CFG4 classes that fit the simplicity criteria
could be migrated here, but probably better to re-implement 
very focussed converter classes.

#]=]
set(G4_VERBOSE OFF)
find_package(G4         REQUIRED MODULE)
find_package(GGeo       REQUIRED CONFIG)   

# just for X4GDMLWrite
set(OpticksXercesC_VERBOSE OFF)
find_package(OpticksXercesC REQUIRED MODULE)

message(STATUS "extg4/CMakeLists.txt:Geant4_DEFINITIONS:${Geant4_DEFINITIONS} ")


find_package(CLHEP      REQUIRED CONFIG) #   Geant4 should have found it already

if(CLHEP_FOUND)
message(STATUS "extg4/CMakeLists.txt:CLHEP_DIR : ${CLHEP_DIR}" )
message(STATUS "extg4/CMakeLists.txt:CLHEP_INCLUDE_DIRS : ${CLHEP_INCLUDE_DIRS}" )
message(STATUS "extg4/CMakeLists.txt:CLHEP_LIBRARIES    : ${CLHEP_LIBRARIES}" )
else()
message(STATUS "extg4/CMakeLists.txt:CLHEP NOT FOUND")
endif()

set(PMTSim_VERBOSE ON)
find_package(PMTSim_standalone CONFIG)
find_package(PMTFastSim CONFIG)

if(PMTSim_VERBOSE)
if(PMTSim_standalone_FOUND)
   message(STATUS "extg4/CMakeLists.txt:PMTSim_FOUND")
else()
   message(STATUS "extg4/CMakeLists.txt:PMTSim_NOT-FOUND")
endif()
endif()

if(TARGET Opticks::PMTSim_standalone)
    if(PMTSim_VERBOSE)
    message(STATUS "extg4/CMakeLists.txt: TARGET Opticks::PMTSim FOUND ")
    get_target_property(PMTSim_IID Opticks::PMTSim INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(PMTSim_ILL Opticks::PMTSim INTERFACE_LINK_LIBRARIES)
    message(STATUS "extg4/CMakeLists.txt:PMTSim_IID : ${PMTSim_IID}")
    message(STATUS "extg4/CMakeLists.txt:PMTSim_ILL : ${PMTSim_ILL}")
    endif()
else()
    if(PMTSim_VERBOSE)
    message(STATUS "extg4/CMakeLists.txt: TARGET Opticks::PMTSim NOT FOUND ")
    endif()
endif()



set(SOURCES
     
    X4CSG.cc

    X4Gen.cc
    X4.cc
    X4Intersect.cc
    X4Simtrace.cc

    X4SolidMaker.cc
    X4VolumeMaker.cc

    X4Named.cc
    X4NameOrder.cc
    X4Entity.cc
    X4SolidBase.cc
    X4Solid.cc
    X4SolidExtent.cc
    X4Mesh.cc
    X4SolidList.cc
    X4SolidTree.cc

    X4GDMLParser.cc
    X4GDMLWriteStructure.cc
    X4GDMLReadStructure.cc
    X4SolidStore.cc

    X4Material.cc
    X4MaterialWaterStandalone.cc
    X4MaterialWater.cc
    X4OpRayleigh.cc
    X4MaterialTable.cc
    X4MaterialPropertiesTable.cc

    X4MaterialLib.cc
    X4PropertyMap.cc
    X4Property.cc

    X4LogicalBorderSurfaceTable.cc
    X4LogicalSkinSurfaceTable.cc

    X4LogicalBorderSurface.cc
    X4LogicalSkinSurface.cc
    X4LogicalSurface.cc

    X4OpticalSurface.cc

    X4Array.cc
    X4MaterialPropertyVector.cc
    X4PhysicsVector.cc

    X4RotationMatrix.cc
    X4Transform3D.cc
    X4AffineTransform.cc
    X4ThreeVector.cc

    X4Geo.cc
    X4PhysicalVolume.cc
    X4Sample.cc

    X4Scintillation.cc 

    X4OpNoviceMaterials.cc
    OpNoviceDetectorConstruction.cc
    LXe_Materials.cc 


    X4GDML.cc
    X4Dump.cc   
    X4GDMLMatrix.cc

    X4_LOG.cc

)
set(HEADERS

    X4CSG.hh

    X4Gen.hh
    X4.hh
    X4Intersect.hh
    X4Simtrace.hh

    X4SolidMaker.hh
    X4VolumeMaker.hh

    x4geomdefs.h
    x4solid.h


    X4Named.hh
    X4NameOrder.hh
    X4Entity.hh
    X4SolidBase.hh
    X4Solid.hh
    X4SolidExtent.hh
    X4Mesh.hh
    X4SolidList.hh
    X4SolidTree.hh

    X4GDMLParser.hh
    X4GDMLWriteStructure.hh
    X4GDMLReadStructure.hh
    X4SolidStore.hh

    X4Material.hh
    X4MaterialWaterStandalone.hh
    X4MaterialWater.hh
    X4OpRayleigh.hh

    X4MaterialTable.hh
    X4MaterialPropertiesTable.hh

    X4MaterialLib.hh
    X4PropertyMap.hh
    X4Property.hh

    X4LogicalBorderSurfaceTable.hh
    X4LogicalSkinSurfaceTable.hh

    X4LogicalBorderSurface.hh
    X4LogicalSkinSurface.hh

    X4LogicalSurface.hh
    X4OpticalSurface.hh

    X4OpticalSurfaceFinish.hh
    X4OpticalSurfaceModel.hh
    X4SurfaceType.hh

    X4Array.hh
    X4MaterialPropertyVector.hh
    X4PhysicsVector.hh

    X4Transform3D.hh
    X4AffineTransform.hh
    X4ThreeVector.hh
    X4RotationMatrix.hh

    X4Geo.hh
    X4PhysicalVolume.hh
    X4Sample.hh

    X4Scintillation.hh


    X4OpNoviceMaterials.hh
    OpNoviceDetectorConstruction.hh
    LXe_Materials.hh

    X4GDML.hh
    X4Dump.hh
    X4GDMLMatrix.hh

    X4_LOG.hh
    X4_API_EXPORT.hh
    X4_HEAD.hh
    X4_TAIL.hh
)

# note that X4PhysicsOrderedFreeVector.hh is missing 


add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )
target_link_libraries( ${name} PUBLIC 
    Opticks::G4
    Opticks::GGeo
    Opticks::OpticksXercesC
    CLHEP::CLHEP
)

if(PMTFastSim_FOUND)
    target_link_libraries( ${name} PUBLIC Opticks::PMTFastSim )
    target_compile_definitions( ${name} PUBLIC WITH_PMTFASTSIM )
elseif(PMTSim_standalone_FOUND)
    target_link_libraries( ${name} PUBLIC Opticks::PMTSim_standalone )
    target_compile_definitions( ${name} PUBLIC WITH_PMTSIM )
endif()


target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
)
target_compile_definitions( ${name} PUBLIC OPTICKS_X4 )  

#target_compile_definitions( ${name} PUBLIC G4MULTITHREADED G4USE_STD11 ) 
#target_compile_definitions( ${name} PUBLIC G4MULTITHREADED ) 
target_compile_definitions( ${name} PUBLIC G4USE_STD11 )  

#[=[
without G4USE_STD11 in use for compilation of the lib many of the tests fail to link 
when using the shared cvmfs Geant4 external with error:

   undefined reference to `G4MTHepRandom::getTheEngine()'

The error shows up when linking the executables but the problem is with the lib. 
Fragility comes from a big switch in the  Randomize.hh header 

#]=]



install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)


