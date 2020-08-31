cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOptiXGeometryOCtx)
project(${name})
include(OpticksBuildOptions)

if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()



find_package(OKConf      REQUIRED CONFIG)   
find_package(SysRap      REQUIRED CONFIG)
find_package(NPY         REQUIRED CONFIG)
find_package(OptiXRap    REQUIRED CONFIG)

#find_package(OptiX       REQUIRED MODULE)   
find_package(OpticksCUDA REQUIRED MODULE) 

set(SOURCES
    ${name}.cc
)

set(OPTIX_SOURCES
    ${name}.cu
    box.cu
    sphere.cu
)

CUDA_WRAP_SRCS( ${name} PTX _generated_PTX_files ${OPTIX_SOURCES} )
CUDA_WRAP_SRCS( ${name} OBJ _generated_OBJ_files ${SOURCES} )

add_executable( ${name} ${_generated_OBJ_files} ${_generated_PTX_files} ${SOURCES} )

target_link_libraries(${name} PUBLIC 
      Opticks::OptiXRap
      Opticks::OKConf 
      Opticks::SysRap 
      Opticks::NPY
      Opticks::OpticksCUDA 
)

install(FILES ${_generated_PTX_files} DESTINATION installcache/PTX)
install(TARGETS ${name}  DESTINATION lib)

#[=[

Attempt to extend the minimal example of UseOptiXProgramPP 
with usage of geometry and minimal pinhole camera.


#]=]

