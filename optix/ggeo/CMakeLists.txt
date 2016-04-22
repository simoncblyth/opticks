cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name GGeo)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 


include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(Boost REQUIRED)
find_package(GLM REQUIRED)
find_package(Bregex REQUIRED)
find_package(Cfg REQUIRED)
find_package(NPY REQUIRED)
find_package(Opticks REQUIRED)


set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS}
   ${GLM_INCLUDE_DIRS} 

   ${Bregex_INCLUDE_DIRS} 
   ${Cfg_INCLUDE_DIRS} 
   ${NPY_INCLUDE_DIRS}
   ${Opticks_INCLUDE_DIRS}
)

set(LIBRARIES
   ${Boost_LIBRARIES}
   ${Bregex_LIBRARIES}
   ${Cfg_LIBRARIES}
   ${NPY_LIBRARIES}
   ${Opticks_LIBRARIES}
)

set(DEFINITIONS
   ${Boost_DEFINITIONS} 
   ${Bregex_DEFINITIONS} 
   ${Cfg_DEFINITIONS} 
   ${Opticks_DEFINITIONS} 
)

include_directories(${INCLUDE_DIRS})   
add_definitions(${DEFINITIONS})


add_library(${name} SHARED 
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
                GFlags.cc
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

                GColors.cc
                GColorMap.cc
                GTraverse.cc
                GColorizer.cc
                GTreeCheck.cc
                GTreePresent.cc
                GBBoxMesh.cc
                GItemList.cc
                GAttrSeq.cc
                GMeshFixer.cc
                GParts.cc
                GCSG.cc
                GPmt.cc
                GMaker.cc
                GCIE.cc
                GGeoTest.cc
                GGeoTestConfig.cc
)



SET(hdrs 
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
GFlags.hh
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
GColors.hh
GColorMap.hh
GTraverse.hh
GColorizer.hh
GTreeCheck.hh
GTreePresent.hh
GBBoxMesh.hh
GItemList.hh
GAttrSeq.hh
GMeshFixer.hh
GParts.hh
GCSG.hh
GPmt.hh
GMaker.hh
GCIE.hh
GGeoTest.hh
GGeoTestConfig.hh
)


message("GGEO:LIBRARIES:${LIBRARIES}")
target_link_libraries(${name} ${LIBRARIES})

install(TARGETS ${name} LIBRARY DESTINATION lib)
install(FILES ${hdrs} DESTINATION include/${name})

add_subdirectory(tests)


