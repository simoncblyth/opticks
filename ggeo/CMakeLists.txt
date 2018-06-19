cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name GGeo)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(OpticksCore REQUIRED CONFIG)

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
    GPmtLib.cc

    GBnd.cc
    GBndLib.cc
    GParts.cc
    GPmt.cc

    GMeshFixer.cc
    GMeshMaker.cc
    GMesh.cc
    GNode.cc
    GTreePresent.cc

    GVolume.cc
    GVolumeList.cc
    GGeoLib.cc
    GGeoCfg.cc
    GGeoTest.cc
    GMaker.cc
    GBBoxMesh.cc
    GColorizer.cc

    GTree.cc
    GTreeCheck.cc

    GTraverse.cc
    GMergedMesh.cc
    GGeoBase.cc
    GGeo.cc
    GNodeLib.cc
    GMeshLib.cc


    GScene.cc 
    GPropertyMap.cc

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
    GPmtLib.hh

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
    GTreeCheck.hh
    GTreePresent.hh
    GBBoxMesh.hh
    GItemList.hh
    GMeshMaker.hh
    GMeshFixer.hh
    GParts.hh
    GCSG.hh
    GPmt.hh
    GMaker.hh
    GCIE.hh
    GGeoTest.hh
    GNodeLib.hh

    GScene.hh
    GPropertyMap.hh
)


add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>) 
target_link_libraries( ${name} PUBLIC Opticks::OpticksCore )
target_compile_definitions( ${name} PUBLIC OPTICKS_GGEO )
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)


add_subdirectory(tests)



