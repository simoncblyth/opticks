cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name NPY)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

## Targets conjured up from cmake/Modules/Find<Name>.cmake  
## (external projects where I do not control CMakeLists.txt)
find_package(PLog    REQUIRED MODULE)   
find_package(GLM     REQUIRED MODULE)
find_package(OpenMesh REQUIRED MODULE)

## Targets serialized via BCM export to PREFIX/lib/cmake/ 
## (controlled from their CMakeLists.txt)
find_package(BoostRap REQUIRED CONFIG)
find_package(YoctoGL REQUIRED CONFIG) 

## Optional targets
find_package(ImplicitMesher CONFIG) 
find_package(DualContouringSample CONFIG) 

# recently rationalized ocsgbsp- has symbol visibility issues, so for now must skip NCSGBSP
#find_package(CSGBSP CONFIG) 

## TODO 
##     splitoff mesh related things depending on (OpenMesh ImplicitMesher DualContouringSample CSGBSP) 
##     into a separate package, pick a name : OpticksMesh/OKMesh  ?


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

    NParameters.cpp
    Report.cpp

    NLookup.cpp


    NSequence.cpp
    Index.cpp
    Counts.cpp

    NTxt.cpp
    NPropNames.cpp

    


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
    NPolygonizer.cpp

    Nd.cpp
    NXform.cpp

    NTris.cpp
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

    NPlaneFromPoints.cpp


    NHybridMesher.cpp
    Nuv.cpp

    BoundarySnappingT.cc
    NLODConfig.cpp
    NSnapConfig.cpp
    NEmitConfig.cpp
    NEmitPhotonsNPY.cpp
    NRngDiffuse.cpp

    NGeometry.cpp
    Ellipse.cpp

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
NTxt.hpp
NPropNames.hpp
NSpectral.hpp
NState.hpp
NConfigurable.hpp
NParameters.hpp
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
NPolygonizer.hpp
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


)



if(YoctoGL_FOUND)  # this is no-longer optional
   message(STATUS "NPY.YoctoGL_FOUND")
   list(APPEND SOURCES  NGLTF.cpp  NScene.cpp  NSceneConfig.cpp NSDF.cpp N.cpp NJS.cpp NMeta.cpp)
   list(APPEND HEADERS  NGLTF.hpp  NScene.hpp  NSceneConfig.hpp NSDF.hpp N.hpp NJS.hpp NMeta.hpp NYJSON.hpp)
endif()

if(ImplicitMesher_FOUND)
   message(STATUS "NPY.ImplicitMesher_FOUND")
   list(APPEND SOURCES  NImplicitMesher.cpp)
   list(APPEND HEADERS  NImplicitMesher.hpp)
endif()

if(DualContouringSample_FOUND)
   message(STATUS "NPY.DualContouringSample_FOUND")
   list(APPEND SOURCES  NDualContouringSample.cpp NOctools.cpp)
   list(APPEND HEADERS  NDualContouringSample.hpp NOct.hpp NOctools.hpp)
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
target_include_directories( ${name} PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
) 

target_compile_definitions( ${name} PUBLIC OPTICKS_NPY )

target_link_libraries( ${name} 
  PUBLIC 
     Opticks::GLM
     Opticks::SysRap 
     Opticks::BoostRap 
     Opticks::OpenMesh
  PRIVATE
     Opticks::PLog
)

if(YoctoGL_FOUND)
   target_link_libraries( ${name} PUBLIC Opticks::YoctoGL )  # headers included into PUBLIC headers need to be PUBLIC
   target_compile_definitions(${name} PUBLIC OPTICKS_YoctoGL )
endif()

if(ImplicitMesher_FOUND)
   target_link_libraries( ${name} PRIVATE Opticks::ImplicitMesher )
   target_compile_definitions(${name} PUBLIC OPTICKS_ImplicitMesher )
endif()

if(DualContouringSample_FOUND)
   target_link_libraries( ${name} PUBLIC Opticks::DualContouringSample )
   target_compile_definitions(${name} PUBLIC OPTICKS_DualContouringSample )
endif()

if(CSGBSP_FOUND)
   target_link_libraries( ${name} PRIVATE Opticks::CSGBSP )
   target_compile_definitions(${name} PUBLIC OPTICKS_CSGBSP  )
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries( ${name} PRIVATE ssl crypto )   # what needs these exactly ?
endif()


bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)

