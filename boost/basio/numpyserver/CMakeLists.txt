cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name NumpyServer)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)
include(EnvCompilationOptions)


find_package(Boost REQUIRED)
find_package(ZMQ REQUIRED)
find_package(AsioZMQ REQUIRED)
find_package(Cfg REQUIRED)
find_package(GLM REQUIRED)
find_package(NPY REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS} 
   ${ZMQ_INCLUDE_DIRS} 
   ${AsioZMQ_INCLUDE_DIRS}
   ${Cfg_INCLUDE_DIRS}  
   ${NPY_INCLUDE_DIRS}  
   ${GLM_INCLUDE_DIRS}  
)

add_definitions( 
   ${Boost_DEFINITIONS} 
    -DVERBOSE 
)

add_library( ${name}  SHARED
    numpydelegate.cpp
)

set(LIBRARIES
      ${Boost_LIBRARIES}
      ${ZMQ_LIBRARIES} 
      ${Cfg_LIBRARIES}
      ${NPY_LIBRARIES}
)

target_link_libraries( ${name} ${LIBRARIES} )

SET(hdrs 
numpydelegate.hpp
numpydelegateCfg.hpp
numpyserver.hpp
net_manager.hpp
npy_server.hpp
udp_server.hpp
)

install(TARGETS ${name}  DESTINATION lib)
install(FILES ${hdrs}    DESTINATION include/${name})


add_subdirectory(tests)

