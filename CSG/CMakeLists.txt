cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
#cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CSG)
#project(${name} VERSION 0.1.0)
project(${name} VERSION 0.1.0 LANGUAGES CXX CUDA)
include(OpticksBuildOptions NO_POLICY_SCOPE)

#find_package(CUDA REQUIRED)
find_package(CUDAToolkit)

find_package(SysRap REQUIRED)


set(SOURCES)
set(HEADERS)

set(CSG_HEADERS)
set(INTERSECT_HEADERS)

list(APPEND SOURCES
    CSGNode.cc
    CU.cc
    CSGPrim.cc
    CSGSolid.cc
    CSGFoundry.cc
    CSGCopy.cc
    CSGMaker.cc
    CSGImport.cc
    CSGTarget.cc


    CSGScan.cc
    CSGScan.cu

    CSGView.cc
    CSGGrid.cc
    CSGQuery.cc
    CSGGeometry.cc
    CSGDraw.cc
    CSGRecord.cc
    CSGSimtraceRerun.cc
    CSGSimtraceSample.cc
    CSGSimtrace.cc

    CSGDebug_Cylinder.cc

    CSG_LOG.cc
)

#    CSGGenstep.cc
#    CSG_stree_Convert.h

list(APPEND CSG_HEADERS
    CSGNode.h

    CU.h
    CSGPrim.h
    CSGSolid.h

    CSGParams.h

    CSGFoundry.h

    CSGCopy.h
    CSGMaker.h
    CSGImport.h
    CSGTarget.h
    CSGScan.h
    CSGView.h
    CSGGrid.h
    CSGQuery.h
    CSGGeometry.h
    CSGDraw.h
    CSGRecord.h

    CSGEnum.h

    CSGSimtraceRerun.h
    CSGSimtrace.hh
    CSGDebug_Cylinder.hh

    CSG_LOG.hh
    CSG_API_EXPORT.hh
)


#    CSGGenstep.h



list(APPEND INTERSECT_HEADERS
    csg_intersect_leaf.h
    csg_intersect_node.h
    csg_intersect_tree.h

    csg_intersect_leaf_box3.h
    csg_intersect_leaf_convexpolyhedron.h
    csg_intersect_leaf_cylinder.h
    csg_intersect_leaf_disc.h
    csg_intersect_leaf_head.h
    csg_intersect_leaf_hyperboloid.h
    csg_intersect_leaf_infcylinder.h
    csg_intersect_leaf_newcone.h
    csg_intersect_leaf_oldcone.h
    csg_intersect_leaf_oldcylinder.h
    csg_intersect_leaf_phicut.h
    csg_intersect_leaf_plane.h
    csg_intersect_leaf_slab.h
    csg_intersect_leaf_sphere.h
    csg_intersect_leaf_templated.h
    csg_intersect_leaf_thetacut.h
    csg_intersect_leaf_zsphere.h

    csg_robust_quadratic_roots.h
    csg_classify.h
    csg_pack.h
    csg_stack.h
    csg_tranche.h
    csg_error.h
    csg_postorder.h

    f4_stack.h
)


list(APPEND HEADERS
    ${CSG_HEADERS}
    ${INTERSECT_HEADERS}
)


message(STATUS "CMAKE_INSTALL_PREFIX     : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_INSTALL_INCLUDEDIR : ${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "CMAKE_INSTALL_LIBDIR     : ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "CMAKE_BINARY_DIR         : ${CMAKE_BINARY_DIR}")


add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )

#CUDA_ADD_LIBRARY( ${name} ${SOURCES} )

string (REPLACE "," ";" COMPUTE_ARCHITECTURES_STR "${COMPUTE_ARCHITECTURES}")
set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES "${COMPUTE_ARCHITECTURES_STR}")

target_compile_definitions( ${name} PUBLIC OPTICKS_CSG )
target_compile_definitions( ${name} PUBLIC WITH_CONTIGUOUS )


#[=[
DEBUG,DEBUG_RECORD,DEBUG_CYLINDER,DEBUG_CONE preprocessor macros
-------------------------------------------------------------------

* WARNING : these DEBUG flags are exceedingly verbose
* note that nvcc compilation will fail when these are enabled,
  as they are intended for CPU debugging of GPU code.
  Some use std::vector for collecting CSGRecords.

* the flags effect critical parts of CSG intersection algorithm
* flags are intended for CPU debugging of CUDA compatible code
* only use them for tests of single or small numbers of intersections
* DO NOT COMMIT WITH THESE UNCOMMENTED

#]=]

#target_compile_definitions( ${name} PRIVATE DEBUG )
#target_compile_definitions( ${name} PRIVATE DEBUG_RECORD )
#target_compile_definitions( ${name} PRIVATE DEBUG_CYLINDER )
#target_compile_definitions( ${name} PRIVATE DEBUG_CONE )

target_compile_definitions( ${name} PUBLIC WITH_S_BB )


# ${CUDA_INCLUDE_DIRS}

target_include_directories(${name}
     PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${OPTICKS_PREFIX}/externals/glm/glm>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<INSTALL_INTERFACE:externals/glm/glm>
)




#target_link_libraries(${name} PUBLIC ${CUDA_LIBRARIES} Opticks::SysRap)
target_link_libraries(${name} ${CUDA_LIBRARIES} Opticks::SysRap)



set( SCRIPTS
    __init__.py
   CSGFoundry.py
)



bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(PROGRAMS ${SCRIPTS}  DESTINATION py/opticks/CSG)


add_subdirectory(tests)

