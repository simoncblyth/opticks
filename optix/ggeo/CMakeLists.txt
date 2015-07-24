cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name GGeo)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

set(BUILD_SHARED_LIBS ON)

include(EnvBuildOptions)


find_package(Boost REQUIRED)
find_package(Bregex REQUIRED)
find_package(GLM REQUIRED)
find_package(NPY REQUIRED)

find_package(ImGui REQUIRED)


set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS}
   ${Bregex_INCLUDE_DIRS} 
   ${ImGui_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${NPY_INCLUDE_DIRS}
)

set(LIBRARIES
   ${Boost_LIBRARIES}
   ${Bregex_LIBRARIES}
   ${NPY_LIBRARIES}
   ${ImGui_LIBRARIES}
)

set(DEFINITIONS
   ${Boost_DEFINITIONS} 
   ${Bregex_DEFINITIONS} 
   ${ImGui_DEFINITIONS}
   -DGUI_
)

include_directories(${INCLUDE_DIRS})   
add_definitions(${DEFINITIONS})

add_library(${name} SHARED 
                GProperty.cc
                GDomain.cc
                GAry.cc
                GPropertyMap.cc
                GMaterial.cc
                GSkinSurface.cc
                GBorderSurface.cc
                GOpticalSurface.cc
                GBoundary.cc
                GBoundaryLib.cc
                GBoundaryLibMetadata.cc
                GSolid.cc
                GMesh.cc
                GMergedMesh.cc
                GGeo.cc
                GVector.cc
                GMatrix.cc
                GNode.cc
                GBuffer.cc
                GDrawable.cc
                GArray.cc
                GLoader.cc
                GSensor.cc
                GSensorList.cc
                GCache.cc
                GItemIndex.cc
                GMaterialIndex.cc
                GSurfaceIndex.cc
                GFlagIndex.cc
                GColors.cc
                GColorMap.cc
                GTraverse.cc
                GColorizer.cc
                GTreeCheck.cc
)

SET(hdrs 
GGeo.hh
GMaterial.hh
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
GBoundary.hh
GBoundaryLib.hh
GBoundaryLibMetadata.hh
GVector.hh
GMatrix.hh
GNode.hh
GBuffer.hh
GDrawable.hh
GArray.hh
GLoader.hh
GSensor.hh
GSensorList.hh
GCache.hh
GItemIndex.hh
GMaterialIndex.hh
GSurfaceIndex.hh
GFlagIndex.hh
GColors.hh
GColorMap.hh
GTraverse.hh
GColorizer.hh
GTreeCheck.hh
)

target_link_libraries(${name} ${LIBRARIES})

install(TARGETS ${name} LIBRARY DESTINATION lib)
install(FILES ${hdrs} DESTINATION include)

add_subdirectory(tests)


