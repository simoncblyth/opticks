cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OK)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${OptiX_INSTALL_DIR}/SDK/CMake") 

include(EnvBuildOptions)


########### OPTICKS #####

if(WITH_OPTIX)

    include(Macros)
    include(CompilerInfo)
    include(ConfigCompilerFlags)
    string (REPLACE "-pedantic" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

    OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

    if(USING_CLANG_C)
      set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
    endif()

    add_definitions(-DWITH_OPTIX)

    find_package(OptiX ${OPTICKS_OPTIX_VERSION} REQUIRED)
    find_package(CUDA ${OPTICKS_CUDA_VERSION} REQUIRED QUIET)

    find_package(CUDARap REQUIRED)
    find_package(OptiXRap REQUIRED)
    find_package(ThrustRap REQUIRED)
    find_package(OKOP REQUIRED)
    find_package(OpticksGL REQUIRED)

    set(WITH_OPTIX_INCLUDES
       ${CUDA_INCLUDE_DIRS}
       ${CUDARap_INCLUDE_DIRS}  
       ${OptiX_INCLUDE_DIRS}  
       ${OptiXRap_INCLUDE_DIRS}  
       ${ThrustRap_INCLUDE_DIRS}
       ${OKOP_INCLUDE_DIRS} 
       ${OpticksGL_INCLUDE_DIRS} 
    )
    set(WITH_OPTIX_DEFINITIONS
       ${CUDA_DEFINITIONS} 
       ${CUDARap_DEFINITIONS}
       ${OptiX_DEFINITIONS} 
       ${OptiXRap_DEFINITIONS} 
       ${ThrustRap_DEFINITIONS}
       ${OKOP_DEFINITIONS}
       ${OpticksGL_DEFINITIONS}
    )
    set(WITH_OPTIX_LIBRARIES
      ${CUDA_LIBRARIES}
      ${CUDARap_LIBRARIES}
      ${OptiX_LIBRARIES}
      ${OptiXRap_LIBRARIES}
      ${ThrustRap_LIBRARIES}
      ${OKOP_LIBRARIES}
      ${OpticksGL_LIBRARIES}
      ssl
      crypto 
   )

else()
   message("Operating without OPTIX ")
   set(WITH_OPTIX_INCLUDES) 
   set(WITH_OPTIX_DEFINITIONS) 
   set(WITH_OPTIX_LIBRARIES) 
endif()

include(EnvCompilationFlags)   

########### BASE #####

find_package(PLog REQUIRED)
find_package(GLM  REQUIRED)
find_package(OpticksBoost REQUIRED)
find_package(SysRap REQUIRED)
find_package(BoostRap REQUIRED)
find_package(NPY   REQUIRED)
find_package(OpticksCore REQUIRED)

set(BASE_INCLUDES
   ${PLog_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${OpticksBoost_INCLUDE_DIRS} 
   ${SysRap_INCLUDE_DIRS}
   ${BoostRap_INCLUDE_DIRS}
   ${NPY_INCLUDE_DIRS} 
   ${OpticksCore_INCLUDE_DIRS} 
)
set(BASE_LIBRARIES
   ${PLog_LIBRARIES}
   ${GLM_LIBRARIES}
   ${OpticksBoost_LIBRARIES}
   ${SysRap_LIBRARIES}
   ${BoostRap_LIBRARIES}
   ${NPY_LIBRARIES}
   ${OpticksCore_LIBRARIES}
)
set(BASE_DEFINITIONS
   ${PLog_DEFINITIONS} 
   ${GLM_DEFINITIONS} 
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
find_package(OpticksGeometry REQUIRED)

set(GEOMETRY_INCLUDES
   ${Assimp_INCLUDE_DIRS}  
   ${AssimpRap_INCLUDE_DIRS}  
   ${OpenMesh_INCLUDE_DIRS}  
   ${OpenMeshRap_INCLUDE_DIRS}  
   ${GGeo_INCLUDE_DIRS}  
   ${OpticksGeometry_INCLUDE_DIRS}  
)
set(GEOMETRY_LIBRARIES
   ${Assimp_LIBRARIES} 
   ${AssimpRap_LIBRARIES} 
   ${OpenMesh_LIBRARIES} 
   ${OpenMeshRap_LIBRARIES} 
   ${GGeo_LIBRARIES} 
   ${OpticksGeometry_LIBRARIES}  
)
set(GEOMETRY_DEFINITIONS
   ${Assimp_DEFINITIONS}  
   ${AssimpRap_DEFINITIONS}  
   ${OpenMesh_DEFINITIONS}  
   ${OpenMeshRap_DEFINITIONS}  
   ${GGeo_DEFINITIONS}  
   ${OpticksGeometry_DEFINITIONS}  
)

########### OPENGL #####

find_package(OpticksGLEW REQUIRED)
find_package(OpticksGLFW REQUIRED)
find_package(ImGui REQUIRED)
find_package(OGLRap REQUIRED)

set(OPENGL_INCLUDES 
   ${OpticksGLEW_INCLUDE_DIRS} 
   ${OpticksGLFW_INCLUDE_DIRS} 
   ${ImGui_INCLUDE_DIRS}
   ${OGLRap_INCLUDE_DIRS} 
)
set(OPENGL_LIBRARIES 
   ${OpticksGLEW_LIBRARIES}
   ${OpticksGLFW_LIBRARIES}
   ${ImGui_LIBRARIES}
   ${OGLRap_LIBRARIES}
)
set(OPENGL_DEFINITIONS
   ${OpticksGLEW_DEFINITIONS}
   ${OpticksGLFW_DEFINITIONS}
   ${ImGui_DEFINITIONS}
   ${OGLRap_DEFINITIONS}
)


########### NPYSERVER #####

if (WITH_NPYSERVER)
    message("WITH_NPYSERVER") 
    add_definitions(-DWITH_NPYSERVER)

    find_package(ZMQ REQUIRED)  
    find_package(AsioZMQ REQUIRED)
    find_package(NumpyServer REQUIRED)

    set(WITH_NPYSERVER_INCLUDES
       ${ZMQ_INCLUDE_DIRS} 
       ${AsioZMQ_INCLUDE_DIRS} 
       ${NumpyServer_INCLUDE_DIRS} 
    )
    set(WITH_NPYSERVER_DEFINITIONS
      ${ZMQ_DEFINITIONS}
      ${NumpyServer_DEFINITIONS} 
    )
    set(WITH_NPYSERVER_LIBRARIES
      ${ZMQ_LIBRARIES}
      ${NumpyServer_LIBRARIES}
   )
else()
    #message("NOT WITH_NPYSERVER") 
    set(WITH_NPYSERVER_INCLUDES)
    set(WITH_NPYSERVER_DEFINITIONS)
    set(WITH_NPYSERVER_LIBRARIES)
endif()

########### ALL #####

set(INCLUDES
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${BASE_INCLUDES}
   ${GEOMETRY_INCLUDES}
   ${OPENGL_INCLUDES}
   ${WITH_NPYSERVER_INCLUDES}
   ${WITH_OPTIX_INCLUDES}
)
set(DEFINITIONS 
   ${BASE_DEFINITIONS}
   ${GEOMETRY_DEFINITIONS}
   ${OPENGL_DEFINITIONS}
   ${WITH_NPYSERVER_DEFINITIONS}
   ${WITH_OPTIX_DEFINITIONS}
)
set(LIBRARIES 
   ${BASE_LIBRARIES}
   ${GEOMETRY_LIBRARIES}
   ${OPENGL_LIBRARIES}
   ${WITH_NPYSERVER_LIBRARIES}
   ${WITH_OPTIX_LIBRARIES}
)

#message("INCLUDES:${INCLUDES}")

include_directories(${INCLUDES})
add_definitions(${DEFINITIONS})


set(SOURCES

    OK_LOG.cc
    OKMgr.cc 
    OKPropagator.cc 
)
set(HEADERS

    OK_LOG.hh
    OK_API_EXPORT.hh
    OK_HEAD.hh
    OK_TAIL.hh

    OKMgr.hh
    OKPropagator.hh
)


add_library(${name} SHARED  ${SOURCES})
target_link_libraries( ${name} ${LIBRARIES})

install(TARGETS ${name}  DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)

