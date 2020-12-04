cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name NPY)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

## Targets conjured up from cmake/Modules/Find<Name>.cmake  
## (external projects where I do not control CMakeLists.txt)
find_package(PLog    REQUIRED MODULE)   
find_package(GLM     REQUIRED MODULE)

## Targets serialized via BCM export to PREFIX/lib/cmake/ 
## (controlled from their CMakeLists.txt)
find_package(BoostRap REQUIRED CONFIG)
find_package(YoctoGL  REQUIRED CONFIG) 

## Former externals
#find_package(OpenMesh REQUIRED MODULE)
#find_package(ImplicitMesher CONFIG) 
#find_package(DualContouringSample CONFIG) 
#find_package(CSGBSP CONFIG) 

set(OpenMesh_FOUND             NO)
set(ImplicitMesher_FOUND       NO)
set(DualContouringSample_FOUND NO)
set(CSGBSP_FOUND               NO)



set(SOURCES
    NPY_LOG.cc
    NPYConfig.cpp

    foo.cpp
    NBufferSpec.cpp
    NPYBase.cpp
    NPYSpec.cpp
    NPYSpecList.cpp
    NSlice.cpp
    NPY.cpp
    NPYList.cpp
    NPYMeta.cpp
    NLoad.cpp

    GLMPrint.cpp
    GLMFormat.cpp

    ViewNPY.cpp
    MultiViewNPY.cpp

    NState.cpp
    NConfigurable.cpp

    Report.cpp

    NLookup.cpp


    NSequence.cpp
    Index.cpp
    Counts.cpp


    


    Types.cpp
    Typ.cpp

    NCache.cpp

    NSensor.cpp
    NSensorList.cpp

    NGeoTestConfig.cpp
    NGunConfig.cpp
    NScan.cpp
    NScanLine.cpp

    No.cpp
    NNode.cpp
    NNodeSample.cpp
    NNodeCollector.cpp
    NGrid.cpp

    NNodeEnum.cpp
    NNodeDump.cpp
    NNodeDump2.cpp
    NNodePoints.cpp
    NNodeUncoincide.cpp
    NNodeNudger.cpp
    NTreeChopper.cpp
    NNodeCoincidence.cpp
    
    NSolid.cpp
    NTreeJUNO.cpp

    NBox.cpp
    NSlab.cpp
    NCylinder.cpp
    NDisc.cpp
    NCone.cpp
    NZSphere.cpp
    NPlane.cpp
    NConvexPolyhedron.cpp

    NPrism.cpp
    NQuad.cpp
    NSphere.cpp
    NTorus.cpp
    NHyperboloid.cpp
    NCubic.cpp


    NBBox.cpp
    NBBoxEnum.cpp
    NPart.cpp
    NTriangle.cpp
    NBoundingBox.cpp
    NTesselate.cpp

    NTriSource.cpp
    NTrianglesNPY.cpp

    NPlanck.cpp
    NCIE.cpp
    NSpectral.cpp

    G4StepNPY.cpp
    SeqNPY.cpp
    TorchStepNPY.cpp
    FabStepNPY.cpp
    GenstepNPY.cpp
    AxisNPY.cpp
    TrivialCheckNPY.cpp

    SequenceNPY.cpp
    PhotonsNPY.cpp
    NPho.cpp
    NPri.cpp

    RecordsNPY.cpp
    HitsNPY.cpp
    BoundariesNPY.cpp
    MaterialLibNPY.cpp
    DummyPhotonsNPY.cpp
    DummyGenstepsNPY.cpp

    PyMCubes/marchingcubes.cpp
    NMarchingCubesNPY.cpp

    NCSG.cpp
    NCSGData.cpp
    NCSGIntersect.cpp
    NCSGList.cpp
    NOctNodeEnum.cpp
    NOctNode.cpp

    NFieldCache.cpp
    NGenerator.cpp
    NGrid3.cpp
    NField3.cpp
    NFieldGrid3.cpp

    NTreeBuilder.cpp
    NTreePositive.cpp
    NTreeBalance.cpp
    NTreeProcess.cpp
    NTreeAnalyse.cpp
    NTreeTools.cpp

    NGPU.cpp

    NGLMExt.cpp
    NGLMCF.cpp

    Nd.cpp
    NXform.cpp

    NTris.cpp

    NPlaneFromPoints.cpp


    Nuv.cpp

    BoundarySnappingT.cc
    NLODConfig.cpp
    NSnapConfig.cpp
    NEmitConfig.cpp
    NEmitPhotonsNPY.cpp
    NRngDiffuse.cpp

    NGeometry.cpp
    Ellipse.cpp
    ImageNPY.cpp

    NPoint.cpp

)

set(_SOURCES
    NOct.cpp
)



SET(HEADERS

NGLM.hpp 
NPY_HEAD.hh
NPY_TAIL.hh
NPY_API_EXPORT.hh
NPY_LOG.hh
NPY_FLAGS.hh
NPart.h

uif.h
ucharfour.h
charfour.h
GLMPrint.hpp
GLMFormat.hpp
NBufferSpec.hpp
NPYBase.hpp
NPY.hpp
NPYList.hpp
NPYMeta.hpp
NLoad.hpp
ViewNPY.hpp
NPYSpec.hpp
NPYSpecList.hpp
MultiViewNPY.hpp
G4StepNPY.hpp
TorchStepNPY.hpp
FabStepNPY.hpp
GenstepNPY.hpp
NGunConfig.hpp
PhotonsNPY.hpp

NPho.hpp
NPri.hpp


HitsNPY.hpp
RecordsNPY.hpp
BoundariesNPY.hpp
MaterialLibNPY.hpp
DummyPhotonsNPY.hpp
DummyGenstepsNPY.hpp


SequenceNPY.hpp
SeqNPY.hpp
AxisNPY.hpp
TrivialCheckNPY.hpp
Counts.hpp
numpy.hpp


NLookup.hpp
Types.hpp
Typ.hpp
Index.hpp

NCache.hpp
NSensor.hpp
NSensorList.hpp
NGeoTestConfig.hpp


NCylinder.h
NSlab.h
NZSphere.h
NPart.h

NSlice.hpp
NSequence.hpp
NBBox.hpp
NBBoxEnum.hpp

No.hpp
NNode.hpp
NNodeSample.hpp
NNodeCollector.hpp


NGrid.hpp

NNodeEnum.hpp
NBox.hpp
NQuad.hpp
NPart.hpp
NBoundingBox.hpp
NTesselate.hpp
NTriangle.hpp

NSlab.hpp
NCylinder.hpp
NDisc.hpp
NCone.hpp
NZSphere.hpp
NSphere.hpp

NTorus.hpp
NHyperboloid.hpp
NCubic.hpp

NPlane.hpp
NConvexPolyhedron.hpp
NPrism.hpp

NPrimitives.hpp
    
NTriSource.hpp
NTrianglesNPY.hpp
NPlanck.hpp
NCIE.hpp
NSpectral.hpp
NState.hpp
NConfigurable.hpp
Report.hpp

PyMCubes/marchingcubes.hpp
NMarchingCubesNPY.hpp


NCSG.hpp
NCSGData.hpp
NCSGIntersect.hpp
NCSGList.hpp
NOctNode.hpp

NFieldCache.hpp
NGenerator.hpp
NGrid3.hpp
NField3.hpp
NFieldGrid3.hpp

NNodeNudger.hpp
NTreeChopper.hpp
NNodeCoincidence.hpp

NSolid.hpp
NTreeJUNO.hpp


NTreeBuilder.hpp
NTreePositive.hpp
NTreeBalance.hpp
NTreeProcess.hpp
NTreeAnalyse.hpp
NTreeTools.hpp

NGPU.hpp


NGLMExt.hpp
NGLMCF.hpp

NSnapConfig.hpp
NLODConfig.hpp

Nuv.hpp
Nd.hpp
NXform.hpp
    
NTris.hpp

NEmitPhotonsNPY.hpp

NGeometry.hpp
Ellipse.hpp
ImageNPY.hpp

NPoint.hpp

)




if(OpenMesh_FOUND)
   list(APPEND SOURCES 
    NOpenMesh.cpp
    NOpenMeshDesc.cpp
    NOpenMeshBoundary.cpp
    NOpenMeshSubdiv.cpp
    NOpenMeshFind.cpp
    NOpenMeshProp.cpp
    NOpenMeshBuild.cpp
    NOpenMeshEnum.cpp
    NOpenMeshCfg.cpp
    NOpenMeshBisect.cpp
    NOpenMeshZipper.cpp

    NPolygonizer.cpp
    NHybridMesher.cpp
    )

    list(APPEND HEADERS
    NPolygonizer.hpp
    )

   target_compile_definitions( ${name} PUBLIC WITH_OPENMESH )

endif()




if(YoctoGL_FOUND)  # this is no-longer optional
   message(STATUS "NPY.YoctoGL_FOUND")
   list(APPEND SOURCES  NGLTF.cpp  NScene.cpp  NSceneConfig.cpp NSDF.cpp N.cpp NJS.cpp NMeta.cpp)
   list(APPEND HEADERS  NGLTF.hpp  NScene.hpp  NSceneConfig.hpp NSDF.hpp N.hpp NJS.hpp NMeta.hpp NYJSON.hpp)
endif()

if(ImplicitMesher_FOUND)
   message(STATUS "NPY.ImplicitMesher_FOUND")
   list(APPEND SOURCES  NImplicitMesher.cpp)
   list(APPEND HEADERS  NImplicitMesher.hpp)
else()
   message(STATUS "NPY.ImplicitMesher_NOT_FOUND")
endif()

if(DualContouringSample_FOUND)
   message(STATUS "NPY.DualContouringSample_FOUND")
   list(APPEND SOURCES  NDualContouringSample.cpp NOctools.cpp)
   list(APPEND HEADERS  NDualContouringSample.hpp NOct.hpp NOctools.hpp)
else()
   message(STATUS "NPY.DualContouringSample_NOT_FOUND")
endif()

if(CSGBSP_FOUND)
   message(STATUS "NPY.CSGBSP_FOUND")
   list(APPEND SOURCES  NCSGBSP.cpp)
   list(APPEND HEADERS  NCSGBSP.hpp)
else()
   message(STATUS "NPY.CSGBSP_NOT-FOUND")
endif()




add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )

# needed for PyMCubes/marchingcubes.cpp
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>) 

# some external header inclusion expects the package name prefix, eg OpenMesh YoctoGL : see notes/issues/NPY-remove-deps-OpenMesh-ImplicitMesher.rst
target_include_directories( ${name} PUBLIC ${OPTICKS_PREFIX}/externals/include )

target_compile_definitions( ${name} PUBLIC OPTICKS_NPY )

target_link_libraries( ${name} 
  PUBLIC 
     Opticks::GLM
     Opticks::SysRap 
     Opticks::BoostRap 
     Opticks::PLog
  PRIVATE
)

top_meta_target(GLM_TOPMETA      "GLM"      "${GLM_targets}" )
top_meta_target(OpenMesh_TOPMETA "OpenMesh" "${OpenMesh_targets}" )


if(OpenMesh_FOUND)
   target_link_libraries( ${name} PUBLIC Opticks::OpenMesh )  
   target_compile_definitions(${name} PUBLIC OPTICKS_OpenMesh )
endif()

if(YoctoGL_FOUND)
   target_link_libraries( ${name} PUBLIC Opticks::YoctoGL )  # headers included into PUBLIC headers need to be PUBLIC
   target_compile_definitions(${name} PUBLIC OPTICKS_YoctoGL )
endif()

if(ImplicitMesher_FOUND)
   target_link_libraries( ${name} PUBLIC Opticks::ImplicitMesher )
   target_compile_definitions(${name} PUBLIC OPTICKS_ImplicitMesher )
endif()

if(DualContouringSample_FOUND)
   target_link_libraries( ${name} PUBLIC Opticks::DualContouringSample )
   target_compile_definitions(${name} PUBLIC OPTICKS_DualContouringSample )
endif()

if(CSGBSP_FOUND)
   target_link_libraries( ${name} PUBLIC Opticks::CSGBSP )
   target_compile_definitions(${name} PUBLIC OPTICKS_CSGBSP  )
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries( ${name} PRIVATE ssl crypto )   # what needs these exactly ?
endif()




set(PC_VERBOSE ON)
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL TOPMATTER "
## NPY TOPMATTER 

${GLM_TOPMETA}
${OpenMesh_TOPMETA}


## end NPY TOPMATTER 
")
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)

