cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(name UseBoostOld)

project(${name})

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)  

set(COMPONENTS
    system 
    program_options 
    filesystem 
    regex 
)

find_package(Boost REQUIRED COMPONENTS ${COMPONENTS})   

message(STATUS "Boost_DIR : ${Boost_DIR} ")
message(STATUS "Boost_INCLUDE_DIRS : ${Boost_INCLUDE_DIRS} ")
message(STATUS "Boost_LIBRARIES    : ${Boost_LIBRARIES} ")

add_executable(${name} ${name}.cc)

target_link_libraries(${name} ${Boost_LIBRARIES})
target_include_directories( ${name} PUBLIC  ${Boost_INCLUDE_DIRS} )

install(TARGETS ${name}  DESTINATION lib)

