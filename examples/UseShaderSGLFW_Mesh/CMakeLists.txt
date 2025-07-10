cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseShaderSGLFW_Mesh)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(OpticksGLFW_ALT ON)
set(OpticksGLFW_VERBOSE ON)

## note this is using sysrap source directory,
## for fast cycle as no need to install sysrap after changes
## BUT just header-only structs not the full shebang library
include_directories($ENV{OPTICKS_HOME}/sysrap)
include_directories($ENV{CUDA_PREFIX}/include)

find_package(GLM         REQUIRED MODULE)
find_package(OpticksGLEW REQUIRED MODULE)
find_package(OpticksGLFW REQUIRED MODULE)
find_package(OpticksCUDA REQUIRED MODULE)
find_package(PLog        REQUIRED MODULE)


message(STATUS "OpticksGLFW_MODULE  : ${OpticksGLFW_MODULE} ")
echo_target_std(Opticks::OpticksGLFW)


set(SOURCES
   ${name}.cc
   SGLFW_CUDA__fillOutputBuffer.cu
)

#[=[
add_executable(${name} ${SOURCES} )
target_link_libraries(${name}
  PUBLIC
    Opticks::GLM
    Opticks::OpticksGLEW
    Opticks::OpticksGLFW
    Opticks::OpticksCUDA
)

target_compile_definitions(${name}
 PUBLIC
     WITH_CUDA_GL_INTEROP
)
#]=]


CUDA_ADD_EXECUTABLE( ${name} ${SOURCES} )
target_link_libraries(${name}
    Opticks::GLM
    Opticks::OpticksGLEW
    Opticks::OpticksGLFW
    Opticks::OpticksCUDA
    Opticks::PLog
)
target_compile_definitions( ${name}
  PUBLIC
    WITH_CUDA_GL_INTEROP
    WITH_CHILD
)


install(TARGETS ${name}  DESTINATION lib)

