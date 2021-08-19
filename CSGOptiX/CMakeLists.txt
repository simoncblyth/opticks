cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CSGOptiX)
project(${name} VERSION 0.1.0)

#[=[
Below extracts BCM hookup for minimal boilerplate package export from the usual "include(OpticksBuildOptions)"
of Opticks packages
#]=]

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")  # override the GNUInstallDirs default of "include"

find_package(BCM CONFIG)

if(NOT BCM_FOUND)
   message(STATUS "CMAKE_MODULE_PATH:${CMAKE_MODULE_PATH}")
   message(STATUS "CMAKE_PREFIX_PATH:${CMAKE_PREFIX_PATH}   expect to find BCM in one of these prefix dirs")
   message(STATUS "typically BCM is found thanks to the CMAKE_PREFIX_PATH envvar including ${OPTICKS_PREFIX}/externals ")
   message(STATUS "see examples/UseBCM to debug failure to find BCM ")
   message(FATAL_ERROR "ABORT " )
endif()

include(BCMDeploy)

set(CMAKE_CXX_STANDARD 11) 
set(CMAKE_CXX_STANDARD_REQUIRED YES)


find_package(CUDA REQUIRED)   
find_package(OpticksCore REQUIRED)
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


include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CSG_INCLUDE_DIRS}
    ${OpticksOptiX_INCLUDE}
    ${OPTICKS_PREFIX}/externals/glm/glm
    ${CMAKE_CURRENT_SOURCE_DIR} 
)

set(COMPUTE_CAPABILITY 30)

set(CUDA_NVCC_FLAGS)
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}")
list(APPEND CUDA_NVCC_FLAGS "-O2")
list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")


set(SOURCES)
set(CU_SOURCES)


list(APPEND SOURCES
    Params.cc 
    CSGOptiX.cc
)

list(APPEND HEADERS
    CSGOptiX.h
)

if(OpticksOptiX_VERSION GREATER_EQUAL 70000)
   list(APPEND CU_SOURCES OptiX7Test.cu)
   list(APPEND SOURCES  
        Frame.cc 
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
   list(APPEND CU_SOURCES OptiX6Test.cu  geo_OptiX6Test.cu )
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

target_link_libraries(${name} 
     PUBLIC 
         ${CUDA_LIBRARIES}
         Opticks::OptiX
         Opticks::CSG
         Opticks::OpticksCore
         Opticks::QUDARap
)
install(FILES ${_generated_PTX_files} DESTINATION ptx)
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

