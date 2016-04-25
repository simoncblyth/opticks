cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name Bregex)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(Boost REQUIRED)

set(LIBRARIES
   ${Boost_LIBRARIES}
)
include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS} 
)
add_definitions(
   ${Boost_DEFINITIONS} 
)

add_library( ${name}  SHARED
   regexsearch.cc
)

target_link_libraries( ${name} 
      ${Boost_LIBRARIES}
)


SET(hdrs 
   regexsearch.hh
)


install(TARGETS ${name} LIBRARY DESTINATION lib)
install(FILES ${hdrs} DESTINATION include/${name})

add_subdirectory(tests EXCLUDE_FROM_ALL)


