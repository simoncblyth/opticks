cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name NumpyServer)
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
find_package(ZMQ REQUIRED)
find_package(AsioZMQ REQUIRED)
find_package(Cfg REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS} 
   ${ZMQ_INCLUDE_DIRS} 
   ${AsioZMQ_INCLUDE_DIRS}
   ${Cfg_INCLUDE_DIRS}  
)


add_definitions( -DVERBOSE )

add_executable( ${name}Test main.cpp  ) 

# mostly header only 
add_library( ${name}  SHARED
    numpydelegate.cpp
)

target_link_libraries( ${name} 
      ${Boost_LIBRARIES}
      ${ZMQ_LIBRARIES} 
      ${Cfg_LIBRARIES}
)

target_link_libraries( ${name}Test 
      ${Boost_LIBRARIES}
      ${ZMQ_LIBRARIES} 
      ${Cfg_LIBRARIES}
      ${name}
)





SET(hdrs 
numpydelegate.hpp
numpydelegateCfg.hpp
numpyserver.hpp
net_manager.hpp
npy_server.hpp
udp_server.hpp
numpy.hpp
)

install(TARGETS ${name}Test DESTINATION bin)
install(TARGETS ${name}     DESTINATION lib)
install(FILES ${hdrs} DESTINATION include)



