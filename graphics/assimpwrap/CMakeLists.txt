cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

set(name AssimpWrap)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 


set(BUILD_SHARED_LIBS ON)

find_package(Assimp REQUIRED)
find_package(GGeo REQUIRED)

include(EnvBuildOptions)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/${name}
    ${Assimp_INCLUDE_DIRS}  
    ${GGeo_INCLUDE_DIRS}  
)

add_definitions(
)

add_library(${name} SHARED 
                src/AssimpGeometry.cc
                src/AssimpCommon.cc
                src/AssimpNode.cc
                src/AssimpTree.cc
                src/AssimpRegistry.cc
                src/AssimpGGeo.cc
                src/AssimpSelection.cc
)

target_link_libraries(${name}
                       ${Assimp_LIBRARIES} 
                       ${GGeo_LIBRARIES} 
                     )

add_executable(${name}Test ${name}Test.cc)

target_link_libraries(${name}Test 
                       ${name}
                       ${Assimp_LIBRARIES} 
                       ${GGeo_LIBRARIES} 
                       ssl
                       crypto 
                     )
                        

install(TARGETS ${name}Test DESTINATION bin)
install(TARGETS ${name} LIBRARY DESTINATION lib)

 

file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/*/*.hh ${CMAKE_CURRENT_SOURCE_DIR}/*/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*/*.hpp )
foreach(abspath ${headers})
   string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" hdr "${abspath}")  # remove the SRC prefix 
   get_filename_component(hbase ${hdr} NAME) 
   get_filename_component(hpath ${hdr} PATH) 
   get_filename_component(hfold ${hpath} NAME) 
   #message( DEBUG " hdr ${hdr} hbase ${hbase} hpath ${hpath} hfold ${hfold}")
   install(FILES ${abspath} DESTINATION include/${hfold})
endforeach(abspath)


