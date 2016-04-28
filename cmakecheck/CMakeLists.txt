cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
# cmakecheck-;cmakecheck-cmake

project(Check)

# have to operate relative to source dir as always use out-of-source builds
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules") 
set(OptiX_INSTALL_DIR "/Developer/OptiX" CACHE PATH "Path to OptiX installed location.")
set(SUPERBUILD ON)
option(DUMP ON)

include(EnvBuildOptions)

find_package(CUDA REQUIRED)
#find_package(OptiX REQUIRED)
#find_package(OpticksBoost REQUIRED)

if(Optix_FOUND)
   message("OptiX_LIBRARIES:${OptiX_LIBRARIES}")
endif(Optix_FOUND)

if(CUDA_FOUND)
   message("CUDA_LIBRARIES:${CUDA_LIBRARIES}")
endif(CUDA_FOUND)




if(DUMP)
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endif(DUMP)




