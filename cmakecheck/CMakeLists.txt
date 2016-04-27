cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
# cmakex-;cmakex-check

project(Check)

# have to operate relative to source dir as always use out-of-source builds
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules") 
set(OptiX_INSTALL_DIR "/Developer/OptiX" CACHE PATH "Path to OptiX installed location.")
set(SUPERBUILD ON)

include(EnvBuildOptions)

find_package(OpticksBoost REQUIRED)

#find_package(OptiX REQUIRED)
#message("OptiX_LIBRARIES:${OptiX_LIBRARIES}")

set(DUMP ON)
if(DUMP)
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endif(DUMP)




