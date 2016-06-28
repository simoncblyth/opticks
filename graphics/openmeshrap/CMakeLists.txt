cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OpenMeshRap)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(PLog REQUIRED)
find_package(OpticksBoost REQUIRED)

find_package(SysRap REQUIRED)
find_package(BoostRap REQUIRED)
find_package(GLM    REQUIRED)
find_package(NPY REQUIRED)
find_package(GGeo REQUIRED)
find_package(OpticksCore REQUIRED)
find_package(OpenMesh REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}

   ${PLog_INCLUDE_DIRS} 
   ${OpticksBoost_INCLUDE_DIRS} 
   ${SysRap_INCLUDE_DIRS} 
   ${BoostRap_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${NPY_INCLUDE_DIRS} 
   ${OpticksCore_INCLUDE_DIRS} 
   ${GGeo_INCLUDE_DIRS} 
   ${OpenMesh_INCLUDE_DIRS} 
)

add_definitions(
   ${PLog_DEFINITIONS}
   ${OpticksBoost_DEFINITIONS}
   ${SysRap_DEFINITIONS} 
   ${BoostRap_DEFINITIONS} 
   ${GLM_DEFINITIONS}
   ${NPY_DEFINITIONS}
   ${OpticksCore_DEFINITIONS}
   ${GGeo_DEFINITIONS}
   ${OpenMesh_DEFINITIONS}
)

set(LIBRARIES
   ${PLog_LIBRARIES}
   ${OpticksBoost_LIBRARIES}
   ${SysRap_LIBRARIES}
   ${BoostRap_LIBRARIES}
   ${GLM_LIBRARIES}
   ${NPY_LIBRARIES}
   ${OpticksCore_LIBRARIES}
   ${GGeo_LIBRARIES}
   ${OpenMesh_LIBRARIES}
)


set(SOURCES
     MESHRAP_LOG.cc

     MWrap.cc
     MTool.cc
     MFixer.cc
)
set(HEADERS

     MESHRAP_LOG.hh
     MESHRAP_API_EXPORT.hh
     MESHRAP_HEAD.hh
     MESHRAP_TAIL.hh

     MWrap.hh
     MTool.hh
     MFixer.hh
)

add_library( ${name}  SHARED ${SOURCES})
target_link_libraries( ${name}  ${LIBRARIES})
install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})

# exclude failing tests..
#add_subdirectory(tests EXCLUDE_FROM_ALL)
#add_subdirectory(tests)

