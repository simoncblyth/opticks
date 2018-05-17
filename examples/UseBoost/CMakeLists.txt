cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(name UseBoost)

project(${name})

#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)  

find_package(Boost REQUIRED COMPONENTS filesystem)   

message(STATUS "Boost_DIR : ${Boost_DIR} ")

add_executable(${name} ${name}.cc)

target_link_libraries(${name} PUBLIC Boost::filesystem)

install(TARGETS ${name}  DESTINATION lib)

