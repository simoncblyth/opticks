cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseGLMSimple)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(GLM_VERBOSE ON)
find_package(GLM REQUIRED MODULE)
add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::GLM )
install(TARGETS ${name}  DESTINATION lib)




