cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name GeoChain)   

#[=[
GeoChain (considered naming it CSG_G4 in analogy to CSG_GGeo)
=================================================================

Raison-d'etre of *GeoChain* is to perform the full chain of geometry conversions in a single executable. 

Note that this means will need to depend on most everything and the kitchensink, but
that is OK as the aim of this package is narrow. The product is the executable not the API. 
See notes in GeoChain.cc.

#]=]


project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(CUDA REQUIRED)  # CUDA headers for vector types like float4, not the CUDA runtime
find_package(CSG_GGeo REQUIRED)
find_package(ExtG4  REQUIRED CONFIG) 

find_package(PMTSim CONFIG)
find_package(jPMTSim CONFIG)   # example of unfound pkg


message(STATUS "CSG_FOUND      : ${CSG_FOUND}      CSG_INCLUDE_DIRS      : ${CSG_INCLUDE_DIRS}")
message(STATUS "GGeo_FOUND     : ${GGeo_FOUND}     GGeo_INCLUDE_DIRS     : ${GGeo_INCLUDE_DIRS}")
message(STATUS "PMTSim_FOUND   : ${PMTSim_FOUND}  ") 
message(STATUS "jPMTSim_FOUND   : ${jPMTSim_FOUND}  ") 

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OPTICKS_PREFIX}/externals/glm/glm
    ${CMAKE_CURRENT_SOURCE_DIR} 
)

set(SOURCES)
set(HEADERS)

list(APPEND SOURCES
   GeoChain.cc
)

list(APPEND HEADERS
   GeoChain.hh
)

add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )

target_link_libraries( ${name} PUBLIC
   Opticks::G4
   Opticks::ExtG4
   Opticks::OpticksXercesC
   Opticks::GGeo
   Opticks::CSG
   Opticks::CSG_GGeo
)

if(PMTSim_FOUND)
   target_link_libraries( ${name} PUBLIC Opticks::PMTSim )
   target_compile_definitions( ${name} PUBLIC WITH_PMTSIM )
endif()

target_compile_definitions( ${name} PUBLIC OPTICKS_GEOCHAIN )

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)



