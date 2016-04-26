cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OpenMeshRap)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(OpticksBoost REQUIRED)
find_package(GLM    REQUIRED)
find_package(NPY REQUIRED)
find_package(GGeo REQUIRED)
find_package(OpticksCore REQUIRED)
find_package(OpenMesh REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}

   ${OpticksBoost_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${NPY_INCLUDE_DIRS} 
   ${OpticksCore_INCLUDE_DIRS} 
   ${GGeo_INCLUDE_DIRS} 
   ${OpenMesh_INCLUDE_DIRS} 
)

add_definitions(
   ${OpenMesh_DEFINITIONS}
   ${GLM_DEFINITIONS}
   ${NPY_DEFINITIONS}
   ${OpticksCore_DEFINITIONS}
   ${GGeo_DEFINITIONS}
   ${OpticksBoost_DEFINITIONS}
)

set(LIBRARIES
   ${OpenMesh_LIBRARIES}
   ${GLM_LIBRARIES}
   ${NPY_LIBRARIES}
   ${OpticksCore_LIBRARIES}
   ${GGeo_LIBRARIES}
   ${OpticksBoost_LIBRARIES}
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

target_link_libraries( ${name}  ${LIBRARIES})

install(TARGETS ${name} LIBRARY DESTINATION lib)
install(FILES ${hdrs} DESTINATION include/${name})

add_subdirectory(tests EXCLUDE_FROM_ALL)

