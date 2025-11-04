cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(name CSGOptiX)
project(${name} VERSION 0.1.0 LANGUAGES CXX CUDA)
include(OpticksBuildOptions)



#[=[
For nanobind check
#]=]
#find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)

find_package(nanobind CONFIG)
message(STATUS "${name} nanobind_ROOT  : ${nanobind_ROOT} ")
message(STATUS "${name} nanobind_FOUND : ${nanobind_FOUND} ")


set(Custom4_VERBOSE ON)
find_package(Custom4 CONFIG)

find_package(CUDAToolkit)

find_package(SysRap REQUIRED)
find_package(QUDARap REQUIRED)
find_package(CSG REQUIRED)

set(OpticksOptiX_VERBOSE ON)
find_package(OpticksOptiX REQUIRED MODULE)

get_target_property(CSG_INCLUDE_DIRS Opticks::CSG INTERFACE_INCLUDE_DIRECTORIES)


message(STATUS "${name} Custom4_FOUND:${Custom4_FOUND}  " )
message(STATUS "${name} CSG_FOUND     : ${CSG_FOUND} ")
message(STATUS "${name} CSG_INCLUDE_DIRS : ${CSG_INCLUDE_DIRS} ")
message(STATUS "${name} OpticksOptiX_VERSION  : ${OpticksOptiX_VERSION} ")

set(buildenvpath ${CMAKE_CURRENT_BINARY_DIR}/buildenv.sh)
file(WRITE ${buildenvpath} "export OPTIX_VERSION=${OpticksOptiX_VERSION}\n")
message(STATUS "write to buildenvpath ${buildenvpath} ")



set(SOURCES)
set(OPTIX_SOURCES)


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
    CSGOptiXService.h
    CSGOPTIX_API_EXPORT.hh
    CSGOPTIX_LOG.hh
)

list(APPEND OPTIX_SOURCES
    CSGOptiX7.cu
    Check.cu
)


message(STATUS "OPTIX_SOURCES : ${OPTIX_SOURCES} ")



add_library( ${name}_OPTIX OBJECT ${OPTIX_SOURCES} )

set_target_properties( ${name}_OPTIX
    PROPERTIES
         CUDA_PTX_COMPILATION ON
         POSITION_INDEPENDENT_CODE ON
         CUDA_SEPARABLE_COMPILATION ON
)

add_library( ${name} SHARED ${SOURCES} ${HEADERS}     )
add_dependencies( ${name} ${name}_OPTIX )


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




if(Custom4_FOUND)
   list(APPEND CSGOptiX_COMPILE_DEFINITIONS_PUBLIC WITH_CUSTOM4)
endif()


target_compile_definitions( ${name}         PUBLIC  "${CSGOptiX_COMPILE_DEFINITIONS_PUBLIC}" )
target_compile_definitions( ${name}_OPTIX   PUBLIC  "${CSGOptiX_COMPILE_DEFINITIONS_PUBLIC}" )

target_compile_definitions( ${name}         PRIVATE "${CSGOptiX_COMPILE_DEFINITIONS_PRIVATE}" )
target_compile_definitions( ${name}_OPTIX   PRIVATE "${CSGOptiX_COMPILE_DEFINITIONS_PRIVATE}" )

set_target_properties(${name}_OPTIX PROPERTIES CUDA_ARCHITECTURES "${COMPUTE_CAPABILITY}")  ## single integer CC of oldest GPU

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  ## write compile_commands.json into build dir

target_compile_options( ${name}_OPTIX
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
      ${CUDA_INCLUDE_DIRS}
#]=]


target_include_directories(${name}
   PUBLIC
      $<BUILD_INTERFACE:${OPTICKS_PREFIX}/externals/glm/glm>
      $<INSTALL_INTERFACE:externals/glm/glm>
      ${OpticksOptiX_INCLUDE}
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_include_directories(${name}_OPTIX
   PUBLIC
      ${OpticksOptiX_INCLUDE}
      $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
      $<BUILD_INTERFACE:${OPTICKS_PREFIX}/include/SysRap>
      $<BUILD_INTERFACE:${OPTICKS_PREFIX}/include/CSG>
      $<BUILD_INTERFACE:${OPTICKS_PREFIX}/include/QUDARap>
)

if(Custom4_FOUND)
   target_include_directories( ${name}_OPTIX PUBLIC ${Custom4_INCLUDE_DIR})
endif()




target_link_libraries(${name}
     PUBLIC
         CUDA::cudart
         Opticks::OptiX
         Opticks::CSG
         Opticks::QUDARap
)


set(py_ext opticks_${name})
if(nanobind_FOUND)
   nanobind_add_module(${py_ext} NB_STATIC ${py_ext}.cc)
   target_link_libraries(${py_ext} PUBLIC ${name})

   nanobind_add_stub(${py_ext}_stub MODULE ${py_ext} OUTPUT ${py_ext}.pyi PYTHON_PATH ${CMAKE_CURRENT_BINARY_DIR})
   install(TARGETS ${py_ext} LIBRARY DESTINATION py)
   install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${py_ext}.pyi DESTINATION py)
endif()

#[=[
Use from python with: "import opticks_CSGOptiX as cx"
#]=]



set(INSTALLED_SCRIPTS
   cxr_min.sh
   cxr_min_debug.sh
   cxt_min.sh
   cxt_min.py

   cxs_min.sh
   cxs_min.py
   cxs_min_AB.py
   cxs_min_lite.py
)


install( TARGETS ${name}_OPTIX OBJECTS DESTINATION optix )

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(PROGRAMS ${INSTALLED_SCRIPTS} DESTINATION bin)


add_subdirectory(tests)

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)







