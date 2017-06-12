cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name GGeo)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(PLog REQUIRED)
find_package(OpticksBoost REQUIRED)
find_package(GLM REQUIRED)
find_package(SysRap REQUIRED)
find_package(BoostRap REQUIRED)
find_package(NPY REQUIRED)
find_package(OpticksCore REQUIRED)

message("GGEO.NPY_INCLUDE_DIRS : ${NPY_INCLUDE_DIRS}")

set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${PLog_INCLUDE_DIRS}
   ${OpticksBoost_INCLUDE_DIRS}
   ${GLM_INCLUDE_DIRS} 

   ${SysRap_INCLUDE_DIRS} 
   ${BoostRap_INCLUDE_DIRS} 
   ${NPY_INCLUDE_DIRS}
   ${OpticksCore_INCLUDE_DIRS}
)

set(LIBRARIES
   ${PLog_LIBRARIES}
   ${OpticksBoost_LIBRARIES}
   ${SysRap_LIBRARIES}
   ${BoostRap_LIBRARIES}
   ${NPY_LIBRARIES}
   ${OpticksCore_LIBRARIES}
)

set(DEFINITIONS
   ${PLog_DEFINITIONS}
   ${OpticksBoost_DEFINITIONS} 
   ${SysRap_DEFINITIONS} 
   ${BoostRap_DEFINITIONS} 
   ${OpticksCore_DEFINITIONS} 
)

set(SOURCES

    GGEO_LOG.cc

    GColorMap.cc
    GTransforms.cc
    GIds.cc

    GItemList.cc
    GCSG.cc

    GItemIndex.cc

    GBuffer.cc
    GArray.cc
    GMatrix.cc

    GVector.cc

    GDrawable.cc
    GOpticalSurface.cc

    GConstant.cc
    GDomain.cc
    GAry.cc
    GCIE.cc

    GProperty.cc
    GPropertyMap.cc

    GPropertyLib.cc
    GMaterial.cc
    GMaterialLib.cc

    GSkinSurface.cc
    GBorderSurface.cc
    GSurfaceLib.cc

    GSur.cc
    GSurLib.cc

    GScintillatorLib.cc

    GSource.cc
    GSourceLib.cc

    GBndLib.cc
    GParts.cc
    GPmt.cc

    GMeshFixer.cc
    GMesh.cc
    GNode.cc
    GTreePresent.cc

    GSolid.cc
    GGeoLib.cc
    GGeoCfg.cc
    GGeoTest.cc
    GGeoTestConfig.cc
    GMaker.cc
    GBBoxMesh.cc
    GColorizer.cc

    GTreeCheck.cc

    GTraverse.cc
    GMergedMesh.cc
    GGeo.cc


)
set(HEADERS

    GGEO_LOG.hh
    GGEO_API_EXPORT.hh
    GGEO_HEAD.hh
    GGEO_TAIL.hh

    GConstant.hh
)

SET(ALL_HEADERS
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
    GSur.hh
    GSurLib.hh


    GVector.hh
    GMatrix.hh
    GIds.hh
    GTransforms.hh
    GNode.hh
    GBuffer.hh
    GDrawable.hh
    GArray.hh
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

if(YoctoGL_FOUND)
    message("GGEO.YoctoGL_FOUND")
    list(APPEND SOURCES GScene.cc)
else()
    message("GGEO.YoctoGL_NOT_FOUND")
endif()




include_directories(${INCLUDE_DIRS})   
add_definitions(${DEFINITIONS})
add_library(${name} SHARED ${SOURCES})
target_link_libraries(${name} ${LIBRARIES})

install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)


