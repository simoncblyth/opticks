cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name GLFWTest)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
                      "$ENV{OPTIX_SDK_DIR}/CMake"
          ) 

include(EnvBuildOptions)


########### using env/cmake/Modules/*.cmake #####

find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(OGLRap REQUIRED)
find_package(Boost REQUIRED)

find_package(ZMQ REQUIRED)    # can cmake handle pkg depending on other pkgs, to make this simpler ?
find_package(AsioZMQ REQUIRED)
find_package(NumpyServer REQUIRED)
find_package(GLM  REQUIRED)
find_package(Cfg REQUIRED)


############# for OptiX  ##########################
set(BUILD_SHARED_LIBS ON)
include(Macros)
include (CompilerInfo)
include (ConfigCompilerFlags)
OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)
if(USING_CLANG_C)
  set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
endif()
find_package(OptiX REQUIRED)
find_package(CUDA 2.3 REQUIRED)



## c++11 forced by AsioZMQ
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -stdlib=libc++")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif ()




#message(NumpyServer_INCLUDE_DIRS:${NumpyServer_INCLUDE_DIRS})

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
   ${OGLRap_INCLUDE_DIRS} 
   ${Boost_INCLUDE_DIRS} 
   ${ZMQ_INCLUDE_DIRS} 
   ${AsioZMQ_INCLUDE_DIRS} 
   ${NumpyServer_INCLUDE_DIRS} 
   ${Cfg_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
      ${OptiX_INCLUDE} 
      ${CUDA_INCLUDE_DIRS}
)

add_definitions(
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
   ${OGLRap_DEFINITIONS}
   ${OptiX_DEFINITIONS} 
)


#add_executable( GLFWContext glfw_context.cc )
#add_executable( Simple simple.cc )
#
#add_executable( ${name} 
#                main.cc 
#                app.cc
#                numpydelegate.cc
#                Interactor.cc
#                InteractorCfg.cc
#) 
#
#
#target_link_libraries( GLFWContext 
#      ${GLEW_LIBRARIES}
#      ${GLFW_LIBRARIES}
#      ${OGLRap_LIBRARIES}
#)
#target_link_libraries( Simple 
#      ${GLEW_LIBRARIES}
#      ${GLFW_LIBRARIES}
#      ${OGLRap_LIBRARIES}
#)
#
#target_link_libraries( ${name} 
#      ${GLEW_LIBRARIES}
#      ${GLFW_LIBRARIES}
#      ${OGLRap_LIBRARIES}
#      ${Boost_LIBRARIES}
#      ${ZMQ_LIBRARIES}
#      ${Cfg_LIBRARIES}
#)
#install(TARGETS GLFWContext DESTINATION bin)
#install(TARGETS ${name} DESTINATION bin)
#install(TARGETS Simple DESTINATION bin)



function(add_optix_executable target_name)

    # split arguments into three lists 
    CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

    # Create the rules to build the PTX from the CUDA files.
    CUDA_WRAP_SRCS( ${target_name} PTX generated_files ${source_files} ${cmake_options} OPTIONS ${options} )

    add_executable(${target_name}
        ${source_files}
        ${generated_files}
        ${cmake_options}
    )


    #find_library( SUTIL_LIBRARY
    #          NAMES sutil
    #          PATHS $ENV{OPTIX_SAMPLES_INSTALL_DIR}/lib )
    #
    #message("SUTIL_LIBRARY ${SUTIL_LIBRARY} ")


    target_link_libraries( ${target_name} 
                   #        ${SUTIL_LIBRARY}
                           optix
                           optixu
                  ${GLEW_LIBRARIES}
                  ${GLFW_LIBRARIES}
                  ${OGLRap_LIBRARIES}
                  ${Boost_LIBRARIES}
                  ${ZMQ_LIBRARIES}
                  ${Cfg_LIBRARIES}
                  #         ${AssimpWrap_LIBRARIES} 
                  #         ${GGeo_LIBRARIES} 
                  #         ${CUDAWrap_LIBRARIES}
                           ssl
                           crypto 
                         )

endfunction()


add_optix_executable( ${name}OptiX
                main.cc 
                app.cc
                numpydelegate.cc
                Interactor.cc
                InteractorCfg.cc
) 


install(TARGETS ${name}OptiX DESTINATION bin)




