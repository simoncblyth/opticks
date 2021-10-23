cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CSG)
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
find_package(SysRap REQUIRED)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OPTICKS_PREFIX}/externals/glm/glm
    ${OPTICKS_PREFIX}/include/SysRap
    ${CMAKE_CURRENT_SOURCE_DIR} 
)

set(SOURCES)
set(HEADERS)

set(CSG_HEADERS)
set(INTERSECT_HEADERS)

list(APPEND SOURCES
    CSGNode.cc
    CU.cc
    CSGPrim.cc
    CSGPrimSpec.cc
    CSGSolid.cc
    CSGFoundry.cc
    CSGMaker.cc
    CSGName.cc
    CSGTarget.cc
    CSGScan.cc
    CSGView.cc
)
    

list(APPEND CSG_HEADERS
    CSGNode.h 

    CU.h
    CSGPrimSpec.h
    CSGPrim.h
    CSGSolid.h


    CSGFoundry.h
    CSGMaker.h
    CSGName.h
    CSGTarget.h
    CSGScan.h
    CSGView.h
    CSGEnum.h

    CSG_LOG.hh 
)

list(APPEND INTERSECT_HEADERS
    csg_intersect_node.h 
    csg_intersect_tree.h 
    csg_robust_quadratic_roots.h
    csg_classify.h
    csg_pack.h
    csg_stack.h
    csg_tranche.h
    csg_error.h
    csg_postorder.h
)

list(APPEND CUDA_HEADERS
    CUDA_CHECK.h
)

list(APPEND HEADERS
    ${CSG_HEADERS}
    ${INTERSECT_HEADERS}
    ${CUDA_HEADERS}
)


message(STATUS "CMAKE_INSTALL_PREFIX     : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_INSTALL_INCLUDEDIR : ${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "CMAKE_INSTALL_LIBDIR     : ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "CMAKE_BINARY_DIR         : ${CMAKE_BINARY_DIR}")


add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )
target_link_libraries(${name} PUBLIC ${CUDA_LIBRARIES} Opticks::SysRap)

target_include_directories(${name}
     PUBLIC
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions( ${name} PUBLIC OPTICKS_CSG )

add_subdirectory(tests)

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})



