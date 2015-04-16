cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name NPY)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)


## c++11 forced by AsioZMQ
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -stdlib=libc++")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif ()


find_package(Boost REQUIRED)
find_package(GLM   REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
)

add_definitions( 
   ${Boost_DEFINITIONS} 
   ${GLM_DEFINITIONS} 
)

add_library( ${name}  SHARED
    NumpyEvt.cpp
)

set(LIBRARIES
      ${Boost_LIBRARIES}
      ${GLM_LIBRARIES}
)

target_link_libraries( ${name} ${LIBRARIES} )

SET(hdrs 
NumpyEvt.hpp
NPY.hpp
VecNPY.hpp
G4StepNPY.hpp
numpy.hpp
)

install(TARGETS ${name}  DESTINATION lib)
install(FILES ${hdrs}    DESTINATION include)

add_subdirectory(tests)

