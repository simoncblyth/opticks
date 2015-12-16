cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name Opticks)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)


find_package(Boost REQUIRED)
find_package(NPY REQUIRED)
find_package(Cfg REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS} 
   ${NPY_INCLUDE_DIRS}  
   ${Cfg_INCLUDE_DIRS}  
)

add_definitions( 
   ${Boost_DEFINITIONS} 
   ${NPY_DEFINITIONS} 
   ${Cfg_DEFINITIONS} 
    -DVERBOSE 
)

add_library( ${name}  SHARED
    Opticks.cc
    OpticksCfg.cc
)

set(LIBRARIES
      ${Boost_LIBRARIES}
      ${Cfg_LIBRARIES}
      ${NPY_LIBRARIES}
)

target_link_libraries( ${name} ${LIBRARIES} )

SET(hdrs 
    Opticks.hh
    OpticksCfg.hh
)

install(TARGETS ${name}  DESTINATION lib)
install(FILES ${hdrs}    DESTINATION include)

add_subdirectory(tests)

