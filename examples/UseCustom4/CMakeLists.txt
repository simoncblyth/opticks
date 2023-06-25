cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseCustom4)
project(${name})

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
   set(CMAKE_CXX_STANDARD 14) 
   set(CMAKE_CXX_STANDARD_REQUIRED on) 
else ()
   #set(CMAKE_CXX_STANDARD 14)
   set(CMAKE_CXX_STANDARD 17)   ## Geant4 1100 forcing c++17 : BUT that restricts to gcc 5+ requiring 
   set(CMAKE_CXX_STANDARD_REQUIRED on) 
endif ()


set(Custom4_VERBOSE ON)

find_package(Custom4 CONFIG)   

message(STATUS "Custom4_DIR : ${Custom4_DIR} ")
message(STATUS "Custom4_FOUND : ${Custom4_FOUND} ")

add_executable(${name} ${name}.cc)

if(Custom4_FOUND)
target_link_libraries(${name} PUBLIC Custom4::Custom4)
endif()

install(TARGETS ${name}  DESTINATION lib)

