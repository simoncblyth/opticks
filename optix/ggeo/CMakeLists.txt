cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name GGeo)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(OpticksBoost REQUIRED)
find_package(GLM REQUIRED)
find_package(BRegex REQUIRED)
find_package(BCfg REQUIRED)
find_package(NPY REQUIRED)
find_package(OpticksCore REQUIRED)


set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${OpticksBoost_INCLUDE_DIRS}
   ${GLM_INCLUDE_DIRS} 

   ${BRegex_INCLUDE_DIRS} 
   ${BCfg_INCLUDE_DIRS} 
   ${NPY_INCLUDE_DIRS}
   ${OpticksCore_INCLUDE_DIRS}
)

set(LIBRARIES
   ${OpticksBoost_LIBRARIES}
   ${BRegex_LIBRARIES}
   ${BCfg_LIBRARIES}
   ${NPY_LIBRARIES}
   ${OpticksCore_LIBRARIES}
)

set(DEFINITIONS
   ${OpticksBoost_DEFINITIONS} 
   ${BRegex_DEFINITIONS} 
   ${BCfg_DEFINITIONS} 
   ${OpticksCore_DEFINITIONS} 
)

include_directories(${INCLUDE_DIRS})   
add_definitions(${DEFINITIONS})


set(SOURCES
    GConstant.cc
    GProperty.cc
    GDomain.cc
    GAry.cc
    GPropertyMap.cc
    GMaterial.cc
    GSource.cc
    GSkinSurface.cc
    GBorderSurface.cc
    GOpticalSurface.cc
    GGeoLib.cc
    GMaterialLib.cc
    GSurfaceLib.cc
    GBndLib.cc
    GScintillatorLib.cc
    GSourceLib.cc
    GPropertyLib.cc
    GSolid.cc
    GMesh.cc
    GMergedMesh.cc
    GGeo.cc
    GGeoCfg.cc
    GVector.cc
    GMatrix.cc
    GIds.cc
    GTransforms.cc
    GNode.cc
    GBuffer.cc
    GDrawable.cc
    GArray.cc
    GCache.cc
    GItemIndex.cc

    GColorMap.cc
    GTraverse.cc
    GColorizer.cc
    GTreeCheck.cc
    GTreePresent.cc
    GBBoxMesh.cc
    GItemList.cc
    GMeshFixer.cc
    GParts.cc
    GCSG.cc
    GPmt.cc
    GMaker.cc
    GCIE.cc
    GGeoTest.cc
    GGeoTestConfig.cc

)


SET(HEADERS
    GConstant.hh
    GGeo.hh
    GGeoCfg.hh
    GMaterial.hh
    GSource.hh
    GMesh.hh
    GMergedMesh.hh
    GProperty.hh
    GDomain.hh
    GAry.hh
    GPropertyMap.hh
    GSolid.hh
    GSkinSurface.hh
    GBorderSurface.hh
    GOpticalSurface.hh
    GGeoLib.hh
    GScintillatorLib.hh
    GSourceLib.hh
    GPropertyLib.hh
    GBndLib.hh
    GMaterialLib.hh
    GSurfaceLib.hh
    GVector.hh
    GMatrix.hh
    GIds.hh
    GTransforms.hh
    GNode.hh
    GBuffer.hh
    GDrawable.hh
    GArray.hh
    GCache.hh
    GItemIndex.hh
    GColorMap.hh
    GTraverse.hh
    GColorizer.hh
    GTreeCheck.hh
    GTreePresent.hh
    GBBoxMesh.hh
    GItemList.hh
    GMeshFixer.hh
    GParts.hh
    GCSG.hh
    GPmt.hh
    GMaker.hh
    GCIE.hh
    GGeoTest.hh
    GGeoTestConfig.hh

)


add_library(${name} SHARED ${SOURCES})
target_link_libraries(${name} ${LIBRARIES})

install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)


