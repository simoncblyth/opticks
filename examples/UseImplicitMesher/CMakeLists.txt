cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseImplicitMesher)
project(${name})

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)  

find_package(ImplicitMesher REQUIRED CONFIG)   

message(STATUS "ImplicitMesher_DIR   : ${ImplicitMesher_DIR} ")
message(STATUS "ImplicitMesher_FOUND : ${ImplicitMesher_FOUND} ")

include(EchoTarget)
echo_target_std(Opticks::ImplicitMesher)

add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::ImplicitMesher)

install(TARGETS ${name}  DESTINATION lib)

