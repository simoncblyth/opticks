cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name BRandom)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

set(BUILD_SHARED_LIBS ON)

include(EnvBuildOptions)


find_package(Boost REQUIRED)
find_package(GLM REQUIRED)
find_package(NPY REQUIRED)

set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
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
                placeholder.cc
)

SET(hdrs 
    placeholder.hh
)

target_link_libraries(${name} ${LIBRARIES})

install(TARGETS ${name} LIBRARY DESTINATION lib)
install(FILES ${hdrs} DESTINATION include)

add_subdirectory(tests)


