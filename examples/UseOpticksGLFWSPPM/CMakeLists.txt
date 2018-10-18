cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOpticksGLFWSPPM)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(OpticksGLFW REQUIRED MODULE)   
find_package(SysRap      REQUIRED CONFIG)   


add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::OpticksGLFW Opticks::SysRap )

install(TARGETS ${name}  DESTINATION lib)

