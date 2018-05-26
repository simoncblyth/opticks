cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOKOP)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(OKOP REQUIRED CONFIG)   

message(STATUS "OKOP_DIR     : ${OKOP_DIR} ")  
message(STATUS "OKOP_VERSION : ${OKOP_VERSION} ")
message(STATUS "OKOP_CONFIG  : ${OKOP_CONFIG} ")

echo_pfx_vars(OKOP "FOUND;VERSION;CONFIG;INCLUDE;LIBRARY;DLL;INSTALL_DIR")
echo_target_std(Opticks::OKOP)

#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::OKOP )
install(TARGETS ${name}  DESTINATION lib)


