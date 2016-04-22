cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OpenMeshRap)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(Boost REQUIRED)
find_package(GLM    REQUIRED)
find_package(NPY REQUIRED)
find_package(GGeo REQUIRED)
find_package(Opticks REQUIRED)
find_package(OpenMesh REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${NPY_INCLUDE_DIRS} 
   ${Opticks_INCLUDE_DIRS} 
   ${GGeo_INCLUDE_DIRS} 
   ${OpenMesh_INCLUDE_DIRS} 
)

add_definitions(
   ${OpenMesh_DEFINITIONS}
   ${GLM_DEFINITIONS}
   ${NPY_DEFINITIONS}
   ${Opticks_DEFINITIONS}
   ${GGeo_DEFINITIONS}
   ${Boost_DEFINITIONS}
)

add_library( ${name}  SHARED
     MWrap.cc
     MTool.cc
     MFixer.cc
) 

set(hdrs
     MWrap.hh
     MTool.hh
     MFixer.hh
)


set(LIBRARIES
    ${OpenMesh_LIBRARIES}
    ${GLM_LIBRARIES}
    ${NPY_LIBRARIES}
    ${Opticks_LIBRARIES}
    ${GGeo_LIBRARIES}
    ${Boost_LIBRARIES}
)

target_link_libraries( ${name}  ${LIBRARIES})


install(TARGETS ${name} LIBRARY DESTINATION lib)
install(FILES ${hdrs} DESTINATION include/${name})

add_subdirectory(tests)

