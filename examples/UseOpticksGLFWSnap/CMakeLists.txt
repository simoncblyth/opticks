cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOpticksGLFWSnap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)
find_package(OpticksGLFW REQUIRED MODULE)   
add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::OpticksGLFW )
install(TARGETS ${name}  DESTINATION lib)

