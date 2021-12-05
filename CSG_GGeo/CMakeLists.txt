cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CSG_GGeo)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions NO_POLICY_SCOPE)

find_package(CUDA REQUIRED)  # CSG_GGeo uses CUDA headers for vector types like float4, it does not use CUDA runtime
find_package(CSG REQUIRED)
find_package(GGeo REQUIRED CONFIG)

message(STATUS "CSG_FOUND      : ${CSG_FOUND}      CSG_INCLUDE_DIRS      : ${CSG_INCLUDE_DIRS}")
message(STATUS "GGeo_FOUND     : ${GGeo_FOUND}     GGeo_INCLUDE_DIRS     : ${GGeo_INCLUDE_DIRS}")

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OPTICKS_PREFIX}/externals/glm/glm
    ${CMAKE_CURRENT_SOURCE_DIR} 
)


set(SOURCES)
set(HEADERS)

list(APPEND SOURCES
    ${name}_Convert.cc
    CSG_GGEO_LOG.cc
)

list(APPEND HEADERS
    ${name}_Convert.h
    CSG_GGEO_API_EXPORT.hh
    CSG_GGEO_LOG.hh
)


add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )

add_executable(${name}Test ${name}Test.cc)

target_link_libraries(${name} PUBLIC Opticks::GGeo Opticks::CSG)

target_compile_definitions( ${name} PUBLIC OPTICKS_CSG_GGEO )

target_link_libraries(${name}Test PUBLIC Opticks::GGeo Opticks::CSG ${name})

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})




