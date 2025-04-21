cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CSGOptiX)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(CUDA REQUIRED)
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
)

list(APPEND HEADERS
    CSGOptiX.h
    CSGOPTIX_API_EXPORT.hh
    CSGOPTIX_LOG.hh
)

if(OpticksOptiX_VERSION GREATER_EQUAL 70000)
   list(APPEND CU_SOURCES
        CSGOptiX7.cu
        Check.cu
    )
   list(APPEND SOURCES
        Ctx.cc
        Properties.cc
        GAS.cc
        GAS_Builder.cc
        PIP.cc
        IAS.cc
        IAS_Builder.cc
        SBT.cc
    )
else()
   # TODO: note that have to manually touch .cu in order to pick up changed CSG headers in build, find way to fix this
   list(APPEND CU_SOURCES
        CSGOptiX6.cu
        CSGOptiX6geo.cu
   )
   list(APPEND SOURCES
        Six.cc
   )
endif()

message(STATUS "CU_SOURCES : ${CU_SOURCES} ")


CUDA_WRAP_SRCS( ${name} PTX _generated_PTX_files ${CU_SOURCES} )
CUDA_WRAP_SRCS( ${name} OBJ _generated_OBJ_files ${SOURCES} )

message(STATUS "_generated_OBJ_files ${_generated_OBJ_files}")
message(STATUS "_generated_PTX_files ${_generated_PTX_files}")

add_library( ${name} SHARED ${SOURCES} ${HEADERS}  ${_generated_OBJ_files} ${_generated_PTX_files}   )

target_compile_definitions( ${name} PRIVATE WITH_PRD )  # using Pointer trick means can reduce attrib and payload to 2
target_compile_definitions( ${name} PUBLIC WITH_SIMULATE )
target_compile_definitions( ${name} PUBLIC WITH_SIMTRACE )
target_compile_definitions( ${name} PUBLIC WITH_RENDER )
target_compile_definitions( ${name} PUBLIC OPTICKS_CSGOPTIX )

# target_compile_definitions( ${name} PUBLIC DEBUG_TAG )     ## NOW FROM sysrap/CMakeLists.txt
# target_compile_definitions( ${name} PUBLIC DEBUG_PIDXYZ )  ## NOW FROM sysrap/CMakeLists.txt Debug/Release CONTROL

#[=[

WITH_THRUST compile definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* WITH_THRUST is needed for Custom4 on GPU complex math functions used by qpmt.h

* However Custom4 also used on CPU without WITH_THRUST (eg by junosw), so
  the WITH_THRUST definition must be PRIVATE to avoid interference with other
  Custom4 usage

#]=]

target_compile_definitions( ${name} PRIVATE WITH_THRUST )


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


target_link_libraries(${name}
     PUBLIC
         ${CUDA_LIBRARIES}
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


install(FILES ${_generated_PTX_files} DESTINATION ptx)
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(PROGRAMS ${INSTALLED_SCRIPTS} DESTINATION bin)



add_subdirectory(tests)

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

