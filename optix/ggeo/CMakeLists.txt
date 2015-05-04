cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name GGeo)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

set(BUILD_SHARED_LIBS ON)

include(EnvBuildOptions)


find_package(Boost REQUIRED)
find_package(NPY REQUIRED)

set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS} 
   ${NPY_INCLUDE_DIRS}
)

set(LIBRARIES
   ${Boost_LIBRARIES}
   ${NPY_LIBRARIES}
)

set(DEFINITIONS
   ${Boost_DEFINITIONS} 
)



include_directories(${INCLUDE_DIRS})   
add_definitions(${DEFINITIONS})


add_library(${name} SHARED 
                md5digest.cc
                GProperty.cc
                GDomain.cc
                GAry.cc
                GPropertyMap.cc
                GMaterial.cc
                GSkinSurface.cc
                GBorderSurface.cc
                GSubstance.cc
                GSubstanceLib.cc
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
)

add_executable(${name}Test ${name}Test.cc)

target_link_libraries(${name} ${LIBRARIES})

target_link_libraries(${name}Test ${name})
                        
install(TARGETS ${name}Test DESTINATION bin)
install(TARGETS ${name} LIBRARY DESTINATION lib)

SET(hdrs 
md5digest.hh
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
GSubstance.hh
GSubstanceLib.hh
GVector.hh
GMatrix.hh
GNode.hh
GBuffer.hh
GDrawable.hh
GArray.hh
)

install(FILES ${hdrs} DESTINATION include)

