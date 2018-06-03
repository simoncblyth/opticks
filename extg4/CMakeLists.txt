cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name ExtG4)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

#[=[
ExtG4 : Objectives
=============================

Bring together G4 and lower level Opticks to 
investigate live G4 direct to GLTF exporting.

For example pulling NPY arrays out of a G4VSolid 
to eventually make GMesh/GParts instances.

Dependencies:

SysRap
   yes
NPY
   yes

GGeo
   inevitable eventually

YoctoGLRap
   hmm : maybe not : retain some distance 
   from GLTF stuff ? 


Class requirements for this package

1. simple single objective classes that 
   wrap single G4 instances
2. named after the classes they enhance
3. expose functionaliy via static functions
   where possible

Some CFG4 classes that fit the simplicity criteria
can be migrated here. 

#]=]


find_package(G4         REQUIRED MODULE)

find_package(SysRap     REQUIRED CONFIG) 
find_package(NPY        REQUIRED CONFIG) 
#find_package(GGeo       REQUIRED CONFIG)  start without for a while
find_package(YoctoGLRap REQUIRED CONFIG)

set(SOURCES
     
    X4VSolid.cc

    X4_LOG.cc

)
set(HEADERS

    X4_LOG.hh
    X4_API_EXPORT.hh
    X4_HEAD.hh
    X4_TAIL.hh
)


add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )
target_link_libraries( ${name} PUBLIC 
    Opticks::G4
    Opticks::SysRap
    Opticks::NPY
    Opticks::YoctoGLRap
)

target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
)

target_compile_definitions( ${name} PUBLIC OPTICKS_X4 ) 

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)


