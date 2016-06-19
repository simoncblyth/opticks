cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name NPY)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(OpticksBoost  REQUIRED)
find_package(GLM    REQUIRED)
find_package(BoostRap REQUIRED)
#message("NPY.BoostRap_INCLUDE_DIRS:${BoostRap_INCLUDE_DIRS} ")


include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}

   ${OpticksBoost_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${BoostRap_INCLUDE_DIRS} 
)

add_definitions( 
   ${OpticksBoost_DEFINITIONS} 
   ${GLM_DEFINITIONS} 
   ${BoostRap_DEFINITIONS} 
)

set(LIBRARIES
   ${OpticksBoost_LIBRARIES}
   ${GLM_LIBRARIES}
   ${BoostRap_LIBRARIES}
)

if(UNIX)
   list(APPEND LIBRARIES ssl crypto)
endif(UNIX)



set(SOURCES

    NPYBase.cpp
    NPYSpec.cpp
    NSlice.cpp
    NPY.cpp

    GLMPrint.cpp
    GLMFormat.cpp

    ViewNPY.cpp
    MultiViewNPY.cpp

    NState.cpp
    NConfigurable.cpp

    Parameters.cpp
    Report.cpp

    Lookup.cpp

    Map.cpp

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

    NQuad.cpp
    NSphere.cpp
    NPlane.cpp
    NPart.cpp
    NPrism.cpp
    NTriangle.cpp
    NBoundingBox.cpp
    NTesselate.cpp
    NTrianglesNPY.cpp

    NPlanck.cpp
    NCIE.cpp
    NSpectral.cpp
)

set(REST_SOURCES

    G4StepNPY.cpp
    TorchStepNPY.cpp
    PhotonsNPY.cpp
    HitsNPY.cpp
    RecordsNPY.cpp
    BoundariesNPY.cpp
    MaterialLibNPY.cpp
    SequenceNPY.cpp
    AxisNPY.cpp
)

SET(HEADERS

NGLM.hpp 
NPY_HEAD.hh
NPY_TAIL.hh
NPY_API_EXPORT.hh

uif.h
ucharfour.h
charfour.h
GLMPrint.hpp
GLMFormat.hpp
NPYBase.hpp
NPY.hpp
ViewNPY.hpp
NPYSpec.hpp
MultiViewNPY.hpp
G4StepNPY.hpp
TorchStepNPY.hpp
NGunConfig.hpp
PhotonsNPY.hpp
HitsNPY.hpp
RecordsNPY.hpp
BoundariesNPY.hpp
MaterialLibNPY.hpp
SequenceNPY.hpp
AxisNPY.hpp
Counts.hpp
numpy.hpp


Lookup.hpp
Types.hpp
Typ.hpp
Index.hpp
Map.hpp
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
Parameters.hpp
)



add_library( ${name}  SHARED ${SOURCES} )
target_link_libraries( ${name} ${LIBRARIES} )
install(TARGETS ${name}  DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})
#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)

