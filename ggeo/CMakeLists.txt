cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name GGeo)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(OpticksCore REQUIRED CONFIG)

#find_package(YoctoGLRap REQUIRED CONFIG)
set(YoctoGLRap_FOUND NO)


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
    GBBox.cc

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

    GScintillatorLib.cc

    GSource.cc
    GSourceLib.cc

    GBnd.cc
    GBndLib.cc
    GParts.cc
    GPts.cc
    GPt.cc

    GMeshFixer.cc
    GMeshMaker.cc
    GMesh.cc
    GNode.cc
    GTreePresent.cc

    GVolume.cc
    GVolumeList.cc
    GGeoLib.cc
    GGeoCfg.cc
    GSolidRec.cc

    GGeoTest.cc
    GMaker.cc
    GBBoxMesh.cc
    GColorizer.cc

    GTree.cc
    GInstancer.cc

    GTraverse.cc
    GMergedMesh.cc
    GGeoBase.cc
    GGeo.cc
    GNodeLib.cc
    GMeshLib.cc


    GPropertyMap.cc
    GPho.cc

)

set(FORMER_SOURCES

    GGeoSensor.cc
    GScene.cc 
    GPmtLib.cc
    GPmt.cc
)

set(FORMER_HEADERS

    GGeoSensor.hh
    GScene.hh
    GPmtLib.hh
    GPmt.hh

)


set(HEADERS

    GGEO_LOG.hh
    GGEO_API_EXPORT.hh
    GGEO_HEAD.hh
    GGEO_TAIL.hh

    GConstant.hh
    GGeoBase.hh
    GGeo.hh
    GGeoCfg.hh
    GSolidRec.hh

    GMaterial.hh
    GSource.hh
    GMesh.hh
    GMergedMesh.hh
    GProperty.hh
    GDomain.hh
    GAry.hh
    GPropertyMap.hh
    GVolume.hh

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
    GBBox.hh
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
    GInstancer.hh
    GTreePresent.hh
    GBBoxMesh.hh
    GItemList.hh
    GMeshMaker.hh
    GMeshFixer.hh
    GParts.hh
    GPts.hh
    GPt.hh
    GCSG.hh
    GMaker.hh
    GCIE.hh
    GGeoTest.hh
    GNodeLib.hh
    GMeshLib.hh

    GPropertyMap.hh
    GPho.hh
)

if(YoctoGLRap_FOUND)
   message(STATUS "GGeo.YoctoGLRap_FOUND")
   list(APPEND SOURCES GGeoGLTF.cc)
   list(APPEND HEADERS GGeoGLTF.hh)
else()
   message(STATUS "GGeo.NOT YoctoGLRap_FOUND")
endif()



add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>) 
target_link_libraries( ${name} PUBLIC Opticks::OpticksCore)

if(YoctoGLRap_FOUND)
target_link_libraries( ${name} PUBLIC Opticks::YoctoGLRap)
target_compile_definitions(${name} PUBLIC WITH_YOCTOGLRAP)
endif()



target_compile_definitions( ${name} PUBLIC OPTICKS_GGEO )
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
add_subdirectory(tests)

