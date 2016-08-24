cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name OpticksGeometry)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)   

find_package(GLM REQUIRED)
find_package(PLog REQUIRED)
find_package(OpticksBoost REQUIRED)

find_package(SysRap REQUIRED)
find_package(BoostRap REQUIRED)
find_package(NPY REQUIRED)
find_package(OpticksCore REQUIRED)

set(BASE_INCLUDES
   ${GLM_INCLUDE_DIRS} 
   ${PLog_INCLUDE_DIRS} 
   ${OpticksBoost_INCLUDE_DIRS} 
   ${SysRap_INCLUDE_DIRS}
   ${BoostRap_INCLUDE_DIRS}
   ${NPY_INCLUDE_DIRS} 
   ${OpticksCore_INCLUDE_DIRS} 
)
set(BASE_LIBRARIES
   ${GLM_LIBRARIES}
   ${PLog_LIBRARIES}
   ${OpticksBoost_LIBRARIES}
   ${SysRap_LIBRARIES}
   ${BoostRap_LIBRARIES}
   ${NPY_LIBRARIES}
   ${OpticksCore_LIBRARIES}
)
set(BASE_DEFINITIONS
   ${GLM_DEFINITIONS} 
   ${PLog_DEFINITIONS} 
   ${OpticksBoost_DEFINITIONS} 
   ${SysRap_DEFINITIONS}
   ${BoostRap_DEFINITIONS}
   ${NPY_DEFINITIONS}
   ${OpticksCore_DEFINITIONS}
)

########### GEOMETRY #####

find_package(Assimp REQUIRED)
find_package(AssimpRap REQUIRED)
find_package(OpenMesh REQUIRED)
find_package(OpenMeshRap REQUIRED)
find_package(GGeo REQUIRED)

set(GEOMETRY_INCLUDES
   ${Assimp_INCLUDE_DIRS}  
   ${AssimpRap_INCLUDE_DIRS}  
   ${OpenMesh_INCLUDE_DIRS}  
   ${OpenMeshRap_INCLUDE_DIRS}  
   ${GGeo_INCLUDE_DIRS}  
)
set(GEOMETRY_LIBRARIES
   ${Assimp_LIBRARIES} 
   ${AssimpRap_LIBRARIES} 
   ${OpenMesh_LIBRARIES} 
   ${OpenMeshRap_LIBRARIES} 
   ${GGeo_LIBRARIES} 
)
set(GEOMETRY_DEFINITIONS
   ${Assimp_DEFINITIONS}  
   ${AssimpRap_DEFINITIONS}  
   ${OpenMesh_DEFINITIONS}  
   ${OpenMeshRap_DEFINITIONS}  
   ${GGeo_DEFINITIONS}  
)

########## ALL #####

set(INCLUDES
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${BASE_INCLUDES}
   ${GEOMETRY_INCLUDES}
)
set(DEFINITIONS 
   ${BASE_DEFINITIONS}
   ${GEOMETRY_DEFINITIONS}
)
set(LIBRARIES 
   ${BASE_LIBRARIES}
   ${GEOMETRY_LIBRARIES}
)

#message("INCLUDES:${INCLUDES}")

include_directories(${INCLUDES})
add_definitions(${DEFINITIONS})


set(SOURCES

    OKGEO_LOG.cc
    OpticksGeometry.cc
    OpticksHub.cc

)
set(HEADERS

    OKGEO_LOG.hh
    OKGEO_API_EXPORT.hh
    OKGEO_HEAD.hh
    OKGEO_TAIL.hh

    OpticksGeometry.hh
    OpticksHub.hh
)



add_library(${name} SHARED ${SOURCES} )
target_link_libraries( ${name} ${LIBRARIES})

install(TARGETS ${name}  DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)

