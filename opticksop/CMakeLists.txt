cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OpticksOp)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
                      "$ENV{OPTIX_SDK_DIR}/CMake"
          ) 

include(EnvBuildOptions)

find_package(OptiX REQUIRED)
find_package(CUDA 2.3 REQUIRED)
find_package(Boost REQUIRED)
find_package(Cfg REQUIRED)
find_package(Opticks REQUIRED)

find_package(GLM REQUIRED)
find_package(NPY REQUIRED)

find_package(OptiXRap REQUIRED)
find_package(CUDAWrap REQUIRED)
find_package(ThrustRap REQUIRED)


include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS}  
   ${NPY_INCLUDE_DIRS}  
   ${Cfg_INCLUDE_DIRS}  
   ${Opticks_INCLUDE_DIRS} 
   ${CUDA_INCLUDE_DIRS}
   ${OptiX_INCLUDE} 
   ${OptiXRap_INCLUDE_DIRS}  
   ${CUDAWrap_INCLUDE_DIRS} 
   ${ThrustRap_INCLUDE_DIRS}
)

add_definitions( 
   ${Boost_DEFINITIONS} 
   ${GLM_DEFINITIONS} 
   ${NPY_DEFINITIONS} 
   ${Cfg_DEFINITIONS} 
   ${OptiX_DEFINITIONS} 
   ${OptiXRap_DEFINITIONS} 
   ${CUDAWrap_DEFINITIONS}
   ${ThrustRap_DEFINITIONS}
   ${Opticks_DEFINITIONS}
    -DVERBOSE 
)

set(LIBRARIES

    optix
    optixu 

   ${Boost_LIBRARIES}
   ${Cfg_LIBRARIES}
   ${Opticks_LIBRARIES}

   ${GLM_LIBRARIES}
   ${NPY_LIBRARIES}

   ${CUDA_LIBRARIES}
   ${OptiXRap_LIBRARIES}
   ${CUDAWrap_LIBRARIES}
   ${ThrustRap_LIBRARIES}

   ssl
   crypto 
)


add_library( ${name}  SHARED
    OpIndexer.cc
    OpIndexerApp.cc
)


target_link_libraries( ${name} ${LIBRARIES} )

SET(hdrs 
    OpIndexer.hh
    OpIndexerApp.hh
)

install(TARGETS ${name}  DESTINATION lib)
install(FILES ${hdrs}    DESTINATION include)

add_subdirectory(tests)

