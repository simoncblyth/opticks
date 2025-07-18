cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseGeometryShader)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(OpticksGLFW_ALT ON)
set(OpticksGLFW_VERBOSE ON)

include_directories($ENV{OPTICKS_HOME}/sysrap)   ## just header-only structs not the full shebang
include_directories(/usr/local/cuda/include)


find_package(GLM         REQUIRED MODULE)
find_package(OpticksGLEW REQUIRED MODULE)
find_package(OpticksGLFW REQUIRED MODULE)
find_package(PLog        REQUIRED MODULE)

#[=[
On Linux note that OpticksGLEW is finding libGLEW.so from ROOT folders.
#]=]


message(STATUS "OpticksGLFW_MODULE  : ${OpticksGLFW_MODULE} ")
echo_target_std(Opticks::OpticksGLFW)

add_executable(${name} ${name}.cc)
target_link_libraries(${name}
  PUBLIC
    Opticks::GLM
    Opticks::OpticksGLEW
    Opticks::OpticksGLFW
    Opticks::PLog
)

target_compile_definitions( ${name}
  PUBLIC
    WITH_CHILD
)


install(TARGETS ${name}  DESTINATION lib)

