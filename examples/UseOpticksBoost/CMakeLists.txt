cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(name UseOpticksBoost)

project(${name})

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)  

#find_package(OpticksBoost REQUIRED CONFIG)   
find_package(OpticksBoost REQUIRED MODULE)   

message(STATUS "OpticksBoost_DIR : ${OpticksBoost_DIR} ")
message(STATUS "OpticksBoost_INCLUDE_DIRS : ${OpticksBoost_INCLUDE_DIRS} ")

add_executable(${name} ${name}.cc)

target_link_libraries(${name} ${OpticksBoost_LIBRARIES})

target_include_directories( ${name} PUBLIC  ${OpticksBoost_INCLUDE_DIRS} )


install(TARGETS ${name}  DESTINATION lib)

