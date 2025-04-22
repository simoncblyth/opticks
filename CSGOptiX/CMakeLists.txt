cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
#cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CSGOptiX)
#project(${name} VERSION 0.1.0)
project(${name} VERSION 0.1.0 LANGUAGES CXX CUDA)
include(OpticksBuildOptions)

#find_package(CUDA REQUIRED)
find_package(CUDAToolkit)

find_package(SysRap REQUIRED)
find_package(QUDARap REQUIRED)
find_package(CSG REQUIRED)

set(OpticksOptiX_VERBOSE ON)
find_package(OpticksOptiX REQUIRED MODULE)

get_target_property(CSG_INCLUDE_DIRS Opticks::CSG INTERFACE_INCLUDE_DIRECTORIES)


message(STATUS "CSG_FOUND     : ${CSG_FOUND} ")
message(STATUS "CSG_INCLUDE_DIRS : ${CSG_INCLUDE_DIRS} ")
message(STATUS "OpticksOptiX_VERSION  : ${OpticksOptiX_VERSION} ")

set(buildenvpath ${CMAKE_CURRENT_BINARY_DIR}/buildenv.sh)
file(WRITE ${buildenvpath} "export OPTIX_VERSION=${OpticksOptiX_VERSION}\n")
message(STATUS "write to buildenvpath ${buildenvpath} ")






#set(COMPUTE_CAPABILITY $ENV{OPTICKS_COMPUTE_CAPABILITY})
#set(CUDA_NVCC_FLAGS)
#list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
#list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}")
#list(APPEND CUDA_NVCC_FLAGS "-O2")
#list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
#list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

#[=[
CUDA_NVCC_FLAGS are set in cmake/Modules/OpticksCUDAFlags.cmake
its better to set them in one place
#]=]


set(SOURCES)
set(CU_SOURCES)


list(APPEND SOURCES
    Params.cc
    CSGOptiX.cc
    Frame.cc
    CSGOPTIX_LOG.cc

    Ctx.cc
    Properties.cc
    GAS.cc
    GAS_Builder.cc
    PIP.cc
    IAS.cc
    IAS_Builder.cc
    SBT.cc
)

list(APPEND HEADERS
    CSGOptiX.h
    CSGOPTIX_API_EXPORT.hh
    CSGOPTIX_LOG.hh
)

list(APPEND CU_SOURCES
    CSGOptiX7.cu
    Check.cu
)


message(STATUS "CU_SOURCES : ${CU_SOURCES} ")



add_library( ${name}PTX OBJECT ${CU_SOURCES} )
#set_property( TARGET ${name}PTX PROPERTY CUDA_PTX_COMPILATION ON)
#set_property( TARGET ${name}PTX PROPERTY POSITION_INDEPENDENT_CODE ON)

set_target_properties( ${name}PTX 
    PROPERTIES
         CUDA_PTX_COMPILATION ON
         POSITION_INDEPENDENT_CODE ON
         CUDA_SEPARABLE_COMPILATION ON
)


#[=[
#CUDA_WRAP_SRCS( ${name} PTX _generated_PTX_files ${CU_SOURCES} )
#CUDA_WRAP_SRCS( ${name} OBJ _generated_OBJ_files ${SOURCES} )
#message(STATUS "_generated_OBJ_files ${_generated_OBJ_files}")
#message(STATUS "_generated_PTX_files ${_generated_PTX_files}")
#add_library( ${name} SHARED ${SOURCES} ${HEADERS}  ${_generated_OBJ_files} ${_generated_PTX_files}   )
#]=]

add_library( ${name} SHARED ${SOURCES} ${HEADERS}     )
add_dependencies( ${name} ${name}PTX )


## CAUTION : DUPLICATION BETWEEN HERE AND sysrap/CMakeLists.txt
set(CSGOptiX_COMPILE_DEFINITIONS_PUBLIC

    $<$<CONFIG:Debug>:CONFIG_Debug>
    $<$<CONFIG:RelWithDebInfo>:CONFIG_RelWithDebInfo>
    $<$<CONFIG:Release>:CONFIG_Release>
    $<$<CONFIG:MinSizeRel>:CONFIG_MinSizeRel>
    $<$<CONFIG:Debug>:DEBUG_PIDX>
    $<$<CONFIG:Debug>:DEBUG_PIDXYZ>
    $<$<CONFIG:Release>:PRODUCTION>

    RNG_PHILOX

    WITH_SIMULATE
    WITH_SIMTRACE
    WITH_RENDER
    OPTICKS_CSGOPTIX
)

set(CSGOptiX_COMPILE_DEFINITIONS_PRIVATE

    WITH_PRD
    WITH_THRUST
)

target_compile_definitions( ${name}      PUBLIC  "${CSGOptiX_COMPILE_DEFINITIONS_PUBLIC}" )
target_compile_definitions( ${name}PTX   PUBLIC  "${CSGOptiX_COMPILE_DEFINITIONS_PUBLIC}" )

target_compile_definitions( ${name}      PRIVATE "${CSGOptiX_COMPILE_DEFINITIONS_PRIVATE}" )
target_compile_definitions( ${name}PTX   PRIVATE "${CSGOptiX_COMPILE_DEFINITIONS_PRIVATE}" )

set_target_properties(${name}PTX PROPERTIES CUDA_ARCHITECTURES "${COMPUTE_CAPABILITY}")  ## single integer CC of oldest GPU

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  ## write compile_commands.json into build dir 

target_compile_options( ${name}PTX 
   PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O2 --use_fast_math -Xcompiler -fPIC>
)


#[=[

CUDA_ARCHITECTURES
--------------------

Note that for ptx generation must specify a single GPU architecture otherwise::

    nvcc fatal   : Option '--ptx (-ptx)' is not allowed when compiling for multiple GPU architectures

Compile Definition Notes
----------------------------


WITH_PRD
~~~~~~~~~

using Pointer trick means can reduce attrib and payload to 2


WITH_THRUST 
~~~~~~~~~~~~

* WITH_THRUST is needed for Custom4 on GPU complex math functions used by qpmt.h

* However Custom4 also used on CPU without WITH_THRUST (eg by junosw), so
  the WITH_THRUST definition must be PRIVATE to avoid interference with other
  Custom4 usage

#]=]



#[=[
      ${CSG_INCLUDE_DIRS}
/usr/local/cuda-12.4/include;/data1/blyth/local/opticks_Debug/include/CSG;/data1/blyth/local/opticks_Debug/externals/glm/glm;/data1/blyth/local/opticks_Debug/include/CSG

      $<BUILD_INTERFACE:/dummy/cx/begin/build>
      $<INSTALL_INTERFACE:/dummy/cx/begin/install>

      $<BUILD_INTERFACE:/dummy/cx/end/build>
      $<INSTALL_INTERFACE:/dummy/cx/end/install>
#]=]


target_include_directories(${name}
   PUBLIC
      $<BUILD_INTERFACE:${OPTICKS_PREFIX}/externals/glm/glm>
      $<INSTALL_INTERFACE:externals/glm/glm>
      ${OpticksOptiX_INCLUDE}
      ${CUDA_INCLUDE_DIRS}
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_include_directories(${name}PTX
   PUBLIC
      ${OpticksOptiX_INCLUDE}
      $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}> 
      $<BUILD_INTERFACE:${OPTICKS_PREFIX}/include/SysRap>
      $<BUILD_INTERFACE:${OPTICKS_PREFIX}/include/CSG>
      $<BUILD_INTERFACE:${OPTICKS_PREFIX}/include/QUDARap>
)


#${CUDA_LIBRARIES}

target_link_libraries(${name}
     PUBLIC
         CUDA::cudart
         Opticks::OptiX
         Opticks::CSG
         Opticks::QUDARap
)


set(INSTALLED_SCRIPTS
   cxr_min.sh
   cxr_min_debug.sh
   cxt_min.sh
   cxt_min.py
)


#install(FILES ${_generated_PTX_files} DESTINATION ptx)
install( TARGETS ${name}PTX OBJECTS DESTINATION ptx )

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(PROGRAMS ${INSTALLED_SCRIPTS} DESTINATION bin)




add_subdirectory(tests)

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

