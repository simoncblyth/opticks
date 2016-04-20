cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
#cmake_policy(SET CMP0054 OLD)
# unfortunately this doesnt suppress the warnings, despite being advertised to do so
# http://www.cmake.org/Wiki/CMake/Policies
set(name GGeoView)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
                      "$ENV{OPTIX_SDK_DIR}/CMake"
          ) 



include(EnvBuildOptions)

set(BUILD_SHARED_LIBS ON)

#####################  for OptiX
include(Macros)
include(CompilerInfo)
include(ConfigCompilerFlags)

OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

if(USING_CLANG_C)
  set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
endif()

OPTION(NPYSERVER "using the numpyserver." OFF)
OPTION(OPTIX     "using OPTIX." OFF)

if (NPYSERVER)
    message("with NPYSERVER") 
    #add_definitions(-DNPYSERVER)
else()
    message("without NPYSERVER") 
endif()


if(OPTIX)
    find_package(OptiX REQUIRED)

    add_definitions(-DWITH_OPTIX)
    find_package(CUDA 2.3 REQUIRED)

    ## c++11 forced by AsioZMQ   needs to come after OptiX
    if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -stdlib=libc++")
    else ()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x")
    endif ()

    find_package(OptiXRap REQUIRED)
    find_package(CUDAWrap REQUIRED)
    find_package(ThrustRap REQUIRED)
    find_package(OpticksOp REQUIRED)
    find_package(OpticksGL REQUIRED)


    set(OPTIONAL_INCLUDE_DIRS

       ${CUDA_INCLUDE_DIRS}
       ${OptiX_INCLUDE}  
       ${OptiXRap_INCLUDE_DIRS}  
       ${CUDAWrap_INCLUDE_DIRS}  
       ${ThrustRap_INCLUDE_DIRS}
       ${OpticksOp_INCLUDE_DIRS} 
       ${OpticksGL_INCLUDE_DIRS} 

    )

    set(OPTIONAL_DEFINITIONS

       ${CUDA_DEFINITIONS} 
       ${OptiX_DEFINITIONS} 
       ${OptiXRap_DEFINITIONS} 
       ${CUDAWrap_DEFINITIONS}
       ${ThrustRap_DEFINITIONS}
       ${OpticksOp_DEFINITIONS}
       ${OpticksGL_DEFINITIONS}

    )

    set(OPTIONAL_LIBRARIES

      optix
      optixu

      ${OptiXRap_LIBRARIES}
      ${CUDA_LIBRARIES}
      ${CUDAWrap_LIBRARIES}
      ${ThrustRap_LIBRARIES}
      ${OpticksOp_LIBRARIES}
      ${OpticksGL_LIBRARIES}

      ssl
      crypto 
   )




else()
   message("Operating without OPTIX ")

   set(OPTIONAL_INCLUDE_DIRS) 
   set(OPTIONAL_DEFINITIONS) 
   set(OPTIONAL_LIBRARIES) 

endif()


########### using env/cmake/Modules/*.cmake #####

find_package(Boost REQUIRED)
find_package(GLM  REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(OGLRap REQUIRED)

if (NPYSERVER)
    find_package(ZMQ REQUIRED)  
    find_package(AsioZMQ REQUIRED)
    find_package(NumpyServer REQUIRED)

    set(NPYSERVER_INCLUDE_DIRS
       ${ZMQ_INCLUDE_DIRS} 
       ${AsioZMQ_INCLUDE_DIRS} 
       ${NumpyServer_INCLUDE_DIRS} 
    )
    set(NPYSERVER_DEFINITIONS
      ${ZMQ_DEFINITIONS}
      ${NumpyServer_DEFINITIONS} 
    )
    set(NPYSERVER_LIBRARIES
      ${ZMQ_LIBRARIES}
      ${NumpyServer_LIBRARIES}
   )

else()
    set(NPYSERVER_INCLUDE_DIRS)
    set(NPYSERVER_DEFINITIONS)
    set(NPYSERVER_LIBRARIES)

endif()

find_package(NPY   REQUIRED)
find_package(Cfg REQUIRED)
find_package(Opticks REQUIRED)

find_package(Assimp REQUIRED)
find_package(AssimpWrap REQUIRED)
find_package(OpenMesh REQUIRED)
find_package(OpenMeshRap REQUIRED)
find_package(GGeo REQUIRED)
find_package(ImGui REQUIRED)
find_package(Bregex REQUIRED)


set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
   ${OGLRap_INCLUDE_DIRS} 
   ${Boost_INCLUDE_DIRS} 

   ${NPY_INCLUDE_DIRS} 
   ${Cfg_INCLUDE_DIRS} 
   ${Opticks_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${Assimp_INCLUDE_DIRS}  
   ${AssimpWrap_INCLUDE_DIRS}  
   ${OpenMesh_INCLUDE_DIRS}  
   ${OpenMeshRap_INCLUDE_DIRS}  
   ${GGeo_INCLUDE_DIRS}  
   ${ImGui_INCLUDE_DIRS}
   ${Bregex_INCLUDE_DIRS}

   ${NPYSERVER_INCLUDE_DIRS}
   ${OPTIONAL_INCLUDE_DIRS}
)


set(DEFINITIONS
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
   ${OGLRap_DEFINITIONS}
   ${Assimp_DEFINITIONS}  
   ${AssimpWrap_DEFINITIONS}  
   ${OpenMesh_DEFINITIONS}  
   ${OpenMeshRap_DEFINITIONS}  
   ${Boost_DEFINITIONS} 
   ${ImGui_DEFINITIONS}
   ${Bregex_DEFINITIONS}
   ${Opticks_DEFINITIONS}

   ${NPYSERVER_DEFINITIONS}
   ${OPTIONAL_DEFINITIONS}
)


set(LIBRARIES 

      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${OGLRap_LIBRARIES}
      ${Boost_LIBRARIES}


      ${Cfg_LIBRARIES}
      ${Opticks_LIBRARIES}

      ${NPY_LIBRARIES}

      ${Assimp_LIBRARIES} 
      ${AssimpWrap_LIBRARIES} 
      ${OpenMesh_LIBRARIES} 
      ${OpenMeshRap_LIBRARIES} 

      ${GGeo_LIBRARIES} 
      ${ImGui_LIBRARIES}
      ${Bregex_LIBRARIES}

      ${NPYSERVER_LIBRARIES}
      ${OPTIONAL_LIBRARIES}
)


include_directories(${INCLUDE_DIRS})
add_definitions(${DEFINITIONS})

add_library(${name}Lib SHARED
              App.cc 
)

SET(hdrs 
              App.hh
)

target_link_libraries( ${name}Lib ${LIBRARIES})

add_executable( ${name} main.cc ) 

target_link_libraries( ${name} ${LIBRARIES} ${name}Lib )

install(TARGETS ${name}Lib DESTINATION lib)
install(FILES ${hdrs}      DESTINATION include)
install(TARGETS ${name}    DESTINATION bin)

add_subdirectory(tests)
