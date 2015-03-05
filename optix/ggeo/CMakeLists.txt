cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

set(name GGeo)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 


set(BUILD_SHARED_LIBS ON)


include(EnvBuildOptions)

add_definitions(
)

add_library(${name} SHARED 
                GProperty.cc
                GPropertyMap.cc
                GMaterial.cc
                GSkinSurface.cc
                GBorderSurface.cc
                GSolid.cc
                GMesh.cc
                GGeo.cc
                GVector.cc
                GMatrix.cc
)

add_executable(${name}Test ${name}Test.cc)

target_link_libraries(${name}Test 
                       ${name}
                     )
                        
install(TARGETS ${name}Test DESTINATION bin)
install(TARGETS ${name} LIBRARY DESTINATION lib)


SET(hdrs 
GGeo.hh
GMaterial.hh
GMesh.hh
GProperty.hh
GPropertyMap.hh
GSolid.hh
GSkinSurface.hh
GBorderSurface.hh
GVector.hh
GMatrix.hh
)

install(FILES ${hdrs} DESTINATION include)


