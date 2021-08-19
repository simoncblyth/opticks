cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CSG_GGeo)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(CUDA REQUIRED)   
find_package(CSG REQUIRED)
find_package(GGeo REQUIRED CONFIG)

message(STATUS "CSG_FOUND      : ${CSG_FOUND}      CSG_INCLUDE_DIRS      : ${CSG_INCLUDE_DIRS}")
message(STATUS "GGeo_FOUND     : ${GGeo_FOUND}     GGeo_INCLUDE_DIRS     : ${GGeo_INCLUDE_DIRS}")

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OPTICKS_PREFIX}/externals/glm/glm
    ${CMAKE_CURRENT_SOURCE_DIR} 
)

add_executable(${name} ${name}.cc ${name}_Convert.cc)

target_link_libraries(${name} PUBLIC Opticks::GGeo Opticks::CSG)

install(TARGETS ${name}  DESTINATION lib)

