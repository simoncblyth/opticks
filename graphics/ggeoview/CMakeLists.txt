cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name GGeoView)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)

########### using env/cmake/Modules/*.cmake #####

find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(OGLRap REQUIRED)
find_package(AssimpWrap REQUIRED)
find_package(GGeo REQUIRED)
#find_package(GLM REQUIRED)


include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
   ${OGLRap_INCLUDE_DIRS} 
   ${AssimpWrap_INCLUDE_DIRS} 
   ${GGeo_INCLUDE_DIRS} 
 #  ${GLM_INCLUDE_DIRS} 
)

add_definitions(
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
   ${OGLRap_DEFINITIONS}
   ${AssimpWrap_DEFINITIONS} 
   ${GGeo_DEFINITIONS} 
)

add_executable( ${name} 
                ggeoview.cc 
                GGeoLoader.cc
) 

target_link_libraries( ${name} 
   ${GLEW_LIBRARIES}
   ${GLFW_LIBRARIES}
   ${OGLRap_LIBRARIES}
   ${AssimpWrap_LIBRARIES} 
   ${GGeo_LIBRARIES} 
)

install(TARGETS ${name} DESTINATION bin)




