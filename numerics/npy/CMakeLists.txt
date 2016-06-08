cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name NPY)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(OpticksBoost  REQUIRED)
find_package(GLM    REQUIRED)
find_package(BoostRap REQUIRED)

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
   ssl 
   crypto
)

# appends to flags for specific files : trying to quell warnings from underlying boost code
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
   set_source_files_properties(jsonutil.cpp PROPERTIES COMPILE_FLAGS -Wno-all) 
endif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")


add_library( ${name}  SHARED
    NPYBase.cpp
    NPY.cpp
    ViewNPY.cpp
    NPYSpec.cpp
    MultiViewNPY.cpp
    GLMPrint.cpp
    GLMFormat.cpp

    Lookup.cpp
    G4StepNPY.cpp
    TorchStepNPY.cpp
    NGunConfig.cpp
    PhotonsNPY.cpp
    HitsNPY.cpp
    RecordsNPY.cpp
    BoundariesNPY.cpp
    MaterialLibNPY.cpp
    SequenceNPY.cpp
    AxisNPY.cpp
    Counts.cpp
    Types.cpp
    Typ.cpp
    Index.cpp
    Map.cpp
    Timer.cpp
    Times.cpp
    TimesTable.cpp
    NCache.cpp
    NSensor.cpp
    NSensorList.cpp
    NSlice.cpp
    NSequence.cpp
    NSphere.cpp
    NPlane.cpp
    NQuad.cpp
    NPart.cpp
    NPrism.cpp
    NBoundingBox.cpp
    NTesselate.cpp
    NTriangle.cpp
    NTrianglesNPY.cpp
    NPlanck.cpp
    NCIE.cpp
    NTxt.cpp
    NPropNames.cpp
    NSpectral.cpp
    NState.cpp
    NConfigurable.cpp
    Parameters.cpp
    Report.cpp

    dirutil.cpp
    stringutil.cpp
    jsonutil.cpp
    timeutil.cpp
    md5digest.cpp
)


target_link_libraries( ${name} ${LIBRARIES} )

SET(hdrs 
crossplatform.hh
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
dirutil.hpp
stringutil.hpp
jsonutil.hpp
timeutil.hpp
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
md5digest.hpp
)

install(TARGETS ${name}  DESTINATION lib)
install(FILES ${hdrs} DESTINATION include/${name})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)

