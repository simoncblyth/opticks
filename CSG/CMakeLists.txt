cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CSG)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions NO_POLICY_SCOPE)

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
    CSGGenstep.cc
    CSGTarget.cc
    CSGScan.cc
    CSGView.cc
    CSGGrid.cc
    CSGQuery.cc
    CSGGeometry.cc
    CSGDraw.cc
    CSGRecord.cc

    CSG_LOG.cc
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
    CSGGenstep.h
    CSGTarget.h
    CSGScan.h
    CSGView.h
    CSGGrid.h
    CSGQuery.h
    CSGGeometry.h
    CSGDraw.h
    CSGRecord.h

    CSGEnum.h

    CSG_LOG.hh 
    CSG_API_EXPORT.hh 
)

list(APPEND INTERSECT_HEADERS
    csg_intersect_leaf.h 
    csg_intersect_node.h 
    csg_intersect_tree.h 

    csg_intersect_leaf_thetacut.h 
    csg_intersect_leaf_phicut.h 

    csg_robust_quadratic_roots.h
    csg_classify.h
    csg_pack.h
    csg_stack.h
    csg_tranche.h
    csg_error.h
    csg_postorder.h

    f4_stack.h
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

target_compile_definitions( ${name} PUBLIC OPTICKS_CSG )

# these DEBUG flags are exceedingly verbose : only use for single photons and do not commit with them uncommented
#target_compile_definitions( ${name} PRIVATE DEBUG )
#target_compile_definitions( ${name} PRIVATE DEBUG_RECORD )


target_include_directories(${name}
     PUBLIC
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(${name} PUBLIC ${CUDA_LIBRARIES} Opticks::SysRap)


bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


add_subdirectory(tests)

