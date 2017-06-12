cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name NPY)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(OpticksBoost  REQUIRED)
find_package(GLM    REQUIRED)
find_package(SysRap REQUIRED)
find_package(BoostRap REQUIRED)
find_package(PLog REQUIRED)
find_package(OpenMesh REQUIRED)

message("NPY.OpenMesh_LIBRARIES :${OpenMesh_LIBRARIES} ")


#find_package(NonExisting) 
#message("NPY.NonExisting_FOUND       :${NonExisting_FOUND} ")
#message("NPY.NonExisting_INCLUDE_DIRS:${NonExisting_INCLUDE_DIRS} ")
#message("NPY.NonExisting_LIBRARIES   :${NonExisting_LIBRARIES} ")

find_package(ImplicitMesher) 
find_package(DualContouringSample) 
find_package(YoctoGL) 
find_package(CSGBSP) 

message("NPY.CSGBSP_INCLUDE_DIRS:${CSGBSP_INCLUDE_DIRS} ")
#message("NPY.YoctoGL_INCLUDE_DIRS:${YoctoGL_INCLUDE_DIRS} ")


set(INCLUDE_DIRS  ${CMAKE_CURRENT_SOURCE_DIR}
   ${OpticksBoost_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${PLog_INCLUDE_DIRS} 
   ${SysRap_INCLUDE_DIRS} 
   ${BoostRap_INCLUDE_DIRS} 
   ${OpenMesh_INCLUDE_DIRS} 
)

set(DEFINITIONS 
   ${OpticksBoost_DEFINITIONS} 
   ${GLM_DEFINITIONS} 
   ${PLog_DEFINITIONS} 
   ${SysRap_DEFINITIONS} 
   ${BoostRap_DEFINITIONS} 
   ${OpenMesh_DEFINITIONS}
)

set(LIBRARIES
   ${OpticksBoost_LIBRARIES}
   ${GLM_LIBRARIES}
   ${PLog_LIBRARIES} 
   ${SysRap_LIBRARIES}
   ${BoostRap_LIBRARIES}
   ${OpenMesh_LIBRARIES}
)


if(UNIX)
   list(APPEND LIBRARIES ssl crypto)
endif(UNIX)


if(ImplicitMesher_FOUND)
   list(APPEND INCLUDE_DIRS ${ImplicitMesher_INCLUDE_DIRS})
   list(APPEND DEFINITIONS -DWITH_ImplicitMesher ${ImplicitMesher_DEFINITIONS} )
   list(APPEND LIBRARIES ${ImplicitMesher_LIBRARIES})
endif()

if(DualContouringSample_FOUND)
   list(APPEND INCLUDE_DIRS ${DualContouringSample_INCLUDE_DIRS})
   list(APPEND DEFINITIONS -DWITH_DualContouringSample ${DualContouringSample_DEFINITIONS} )
   list(APPEND LIBRARIES ${DualContouringSample_LIBRARIES})
endif()

if(YoctoGL_FOUND)
   list(APPEND INCLUDE_DIRS ${YoctoGL_INCLUDE_DIRS})
   list(APPEND DEFINITIONS -DWITH_YoctoGL ${YoctoGL_DEFINITIONS} )
   list(APPEND LIBRARIES ${YoctoGL_LIBRARIES})
endif()

if(CSGBSP_FOUND)
   list(APPEND INCLUDE_DIRS ${CSGBSP_INCLUDE_DIRS})
   list(APPEND DEFINITIONS -DWITH_CSGBSP ${CSGBSP_DEFINITIONS})
   list(APPEND LIBRARIES ${CSGBSP_LIBRARIES})
endif()




set(SOURCES
    NPY_LOG.cc

    foo.cpp
    NPYBase.cpp
    NPYSpec.cpp
    NSlice.cpp
    NPY.cpp
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

    Timer.cpp
    Times.cpp
    TimesTable.cpp

    Types.cpp
    Typ.cpp

    NCache.cpp

    NSensor.cpp
    NSensorList.cpp

    NGunConfig.cpp

    NNode.cpp
    NBox.cpp
    NSlab.cpp
    NCylinder.cpp
    NCone.cpp
    NZSphere.cpp
    NPlane.cpp
    NConvexPolyhedron.cpp

    NPrism.cpp

    NQuad.cpp
    NSphere.cpp
    NBBox.cpp
    NPart.cpp
    NTriangle.cpp
    NBoundingBox.cpp
    NTesselate.cpp
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
    RecordsNPY.cpp
    HitsNPY.cpp
    BoundariesNPY.cpp
    MaterialLibNPY.cpp
    DummyPhotonsNPY.cpp

    PyMCubes/marchingcubes.cpp
    NMarchingCubesNPY.cpp

    NCSG.cpp
    NOctNode.cpp

    NFieldCache.cpp
    NGenerator.cpp
    NGrid3.cpp
    NField3.cpp
    NFieldGrid3.cpp

    NTreeTools.cpp
    NGLMExt.cpp
    NPolygonizer.cpp

    Nd.cpp

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

uif.h
ucharfour.h
charfour.h
GLMPrint.hpp
GLMFormat.hpp
NPYBase.hpp
NPY.hpp
NLoad.hpp
ViewNPY.hpp
NPYSpec.hpp
MultiViewNPY.hpp
G4StepNPY.hpp
TorchStepNPY.hpp
FabStepNPY.hpp
GenstepNPY.hpp
NGunConfig.hpp
PhotonsNPY.hpp
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
Timer.hpp
Times.hpp
TimesTable.hpp
NCache.hpp
NSensor.hpp
NSensorList.hpp
NSlice.hpp
NSequence.hpp
NSphere.hpp
NPlane.hpp
NBBox.hpp
NNode.hpp
NBox.hpp
NQuad.hpp
NPart.hpp
NPrism.hpp
NBoundingBox.hpp
NTesselate.hpp
NTriangle.hpp
NTrianglesNPY.hpp
NPlanck.hpp
NCIE.hpp
NTxt.hpp
NPropNames.hpp
NSpectral.hpp
NState.hpp
NConfigurable.hpp
NParameters.hpp

PyMCubes/marchingcubes.hpp
NMarchingCubesNPY.hpp
NCSG.hpp
NOctNode.hpp

NFieldCache.hpp
NGenerator.hpp
NGrid3.hpp
NField3.hpp
NFieldGrid3.hpp

NTreeTools.hpp
NGLMExt.hpp
NPolygonizer.hpp

)


set(_HEADERS
)



if(NonExisting_FOUND)
   list(APPEND SOURCES  NNonExisting.cpp)
endif()

if(ImplicitMesher_FOUND)
   message("NPY.ImplicitMesher_FOUND")
   list(APPEND SOURCES  NImplicitMesher.cpp)
   list(APPEND HEADERS  NImplicitMesher.hpp)
endif()

if(DualContouringSample_FOUND)
   message("NPY.DualContouringSample_FOUND")
   list(APPEND SOURCES  NDualContouringSample.cpp NOctools.cpp)
   list(APPEND HEADERS  NDualContouringSample.hpp NOct.hpp NOctools.hpp)
endif()

if(CSGBSP_FOUND)
   message("NPY.CSGBSP_FOUND")
   list(APPEND SOURCES  NCSGBSP.cpp)
   list(APPEND HEADERS  NCSGBSP.hpp)
endif()

if(YoctoGL_FOUND)
   message("NPY.YoctoGL_FOUND")
   list(APPEND SOURCES  NGLTF.cpp  NScene.cpp)
   list(APPEND HEADERS  NGLTF.hpp  NScene.hpp)
endif()



include_directories( ${INCLUDE_DIRS} )
add_definitions( ${DEFINITIONS} )
add_library( ${name}  SHARED ${SOURCES} )
target_link_libraries( ${name} ${LIBRARIES} )

install(TARGETS ${name}  DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})

# with command :  c_enums_to_python.py NPart.h  > NPart_h.py 
#add_custom_command(
#    COMMAND ${PYTHON_EXECUTABLE} c_enums_to_python.py NPart.h
#    DEPENDS NPart.h
#    OUTPUTS NPart_h.py
#    COMMENT "Generating python enum header ."
#)


#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)

