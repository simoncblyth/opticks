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
set(G4_VERBOSE ON)
find_package(G4         REQUIRED MODULE)
find_package(GGeo       REQUIRED CONFIG)   

# just for X4GDMLWrite
find_package(OpticksXercesC REQUIRED MODULE)



find_package(CLHEP      REQUIRED CONFIG) #   Geant4 should have found it already

if(CLHEP_FOUND)
message(STATUS "CLHEP_DIR : ${CLHEP_DIR}" )
message(STATUS "CLHEP_INCLUDE_DIRS : ${CLHEP_INCLUDE_DIRS}" )
message(STATUS "CLHEP_LIBRARIES    : ${CLHEP_LIBRARIES}" )
else()
message(STATUS "CLHEP NOT FOUND")
endif()






set(SOURCES
     
    X4CSG.cc

    X4Gen.cc
    X4.cc
    X4Named.cc
    X4Entity.cc
    X4SolidBase.cc
    X4Solid.cc
    X4SolidExtent.cc
    X4Mesh.cc
    X4SolidList.cc

    X4GDMLParser.cc
    X4GDMLWriteStructure.cc
    X4GDMLReadStructure.cc
    X4SolidStore.cc

    X4Material.cc
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

    X4PhysicsVector.cc
    X4RotationMatrix.cc
    X4Transform3D.cc
    X4AffineTransform.cc
    X4ThreeVector.cc
    X4PhysicalVolume.cc
    X4Sample.cc

    

    X4OpNoviceMaterials.cc
    OpNoviceDetectorConstruction.cc
    LXe_Materials.cc 
   

    X4_LOG.cc

)
set(HEADERS

    X4CSG.hh

    X4Gen.hh
    X4.hh
    X4Named.hh
    X4Entity.hh
    X4SolidBase.hh
    X4Solid.hh
    X4SolidExtent.hh
    X4Mesh.hh
    X4SolidList.hh

    X4GDMLParser.hh
    X4GDMLWriteStructure.hh
    X4GDMLReadStructure.hh
    X4SolidStore.hh

    X4Material.hh
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

    X4PhysicsVector.hh
    X4Transform3D.hh
    X4AffineTransform.hh
    X4ThreeVector.hh
    X4RotationMatrix.hh
    X4PhysicalVolume.hh
    X4Sample.hh

    X4OpNoviceMaterials.hh
    OpNoviceDetectorConstruction.hh
    LXe_Materials.hh

    X4_LOG.hh
    X4_API_EXPORT.hh
    X4_HEAD.hh
    X4_TAIL.hh
)


add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )
target_link_libraries( ${name} PUBLIC 
    Opticks::G4
    Opticks::GGeo
    Opticks::OpticksXercesC
    CLHEP::CLHEP
)

target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
)

target_compile_definitions( ${name} PUBLIC OPTICKS_X4 ) 

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)


