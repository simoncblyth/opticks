cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OKOP) 
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${OptiX_INSTALL_DIR}/SDK/CMake") 

include(EnvBuildOptions)
include(EnvCompilationFlags)   

string (REPLACE "-pedantic" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

find_package(OptiX ${OPTICKS_OPTIX_VERSION} REQUIRED)
find_package(CUDA ${OPTICKS_CUDA_VERSION} REQUIRED QUIET)

find_package(PLog REQUIRED)
find_package(OpticksBoost REQUIRED)
find_package(GLM REQUIRED)

find_package(SysRap REQUIRED)
find_package(BoostRap REQUIRED)
find_package(OpticksCore REQUIRED)
find_package(GGeo  REQUIRED)
find_package(OpticksGeometry REQUIRED)
find_package(NPY REQUIRED)
find_package(OptiXRap REQUIRED)
find_package(CUDARap REQUIRED)
find_package(ThrustRap REQUIRED)

# NB **NO OpenGL/GLEW/GLFW** 
#    use opticksgl- not opticksop- for viz things 

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
   ${OpticksGeometry_INCLUDE_DIRS} 
   ${CUDA_INCLUDE_DIRS}
   ${OptiX_INCLUDE_DIRS} 
   ${OptiXRap_INCLUDE_DIRS}  
   ${CUDARap_INCLUDE_DIRS} 
   ${ThrustRap_INCLUDE_DIRS}
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
   ${OpticksGeometry_DEFINITIONS} 
   ${CUDA_DEFINITIONS} 
   ${OptiX_DEFINITIONS} 
   ${OptiXRap_DEFINITIONS} 
   ${CUDARap_DEFINITIONS}
   ${ThrustRap_DEFINITIONS}

    -DVERBOSE 
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
   ${OpticksGeometry_LIBRARIES} 
   ${CUDA_LIBRARIES}
   ${OptiX_LIBRARIES}
   ${OptiXRap_LIBRARIES}
   ${CUDARap_LIBRARIES}
   ${ThrustRap_LIBRARIES}
)

if(UNIX)
   set(LIBRARIES ${LIBRARIES} ssl crypto)
endif(UNIX)


set(SOURCES
     
    OKOP_LOG.cc

    OpZeroer.cc
    OpSeeder.cc
    OpIndexer.cc
    OpIndexer_.cu
    OpIndexerApp.cc
    OpEngine.cc
    OpMgr.cc
    OpPropagator.cc
    OpTracer.cc
)

set(HEADERS

    OKOP_LOG.hh
    OKOP_API_EXPORT.hh
    OKOP_HEAD.hh
    OKOP_TAIL.hh

    OpZeroer.hh
    OpSeeder.hh
    OpIndexer.hh
    OpIndexerApp.hh
    OpEngine.hh
    OpPropagator.hh
    OpTracer.hh
    OpMgr.hh
)

CUDA_ADD_LIBRARY( ${name} SHARED ${SOURCES} OPTIONS)
target_link_libraries( ${name} ${LIBRARIES} )

install(TARGETS ${name}  DESTINATION lib)
install(FILES ${HEADERS}   DESTINATION include/${name})
install(FILES ${HEADERS}   DESTINATION include/Opticks)

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)

