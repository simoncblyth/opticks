cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name C4)   # short for CSG_U4
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions NO_POLICY_SCOPE)

find_package(CUDA REQUIRED)  # CUDA headers are needed for vector types like float4, not for the CUDA runtime
find_package(CSG REQUIRED)
find_package(U4 REQUIRED CONFIG)

message(STATUS "CSG_FOUND      : ${CSG_FOUND}      CSG_INCLUDE_DIRS      : ${CSG_INCLUDE_DIRS}")
message(STATUS "U4_FOUND     : ${U4_FOUND}     U4_INCLUDE_DIRS     : ${U4_INCLUDE_DIRS}")

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OPTICKS_PREFIX}/externals/glm/glm
    ${CMAKE_CURRENT_SOURCE_DIR} 
)


set(SOURCES)
set(HEADERS)

list(APPEND SOURCES
    C4_LOG.cc

    C4.cc
    C4Solid.cc
)

list(APPEND HEADERS
    C4_API_EXPORT.hh
    C4_LOG.hh

    C4.hh
    C4Solid.hh
)

add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )

target_link_libraries(${name} PUBLIC Opticks::U4 Opticks::CSG)
target_compile_definitions( ${name} PUBLIC OPTICKS_C4 )

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)


