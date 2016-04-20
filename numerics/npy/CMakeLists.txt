cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name NPY)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)


## c++11 forced by AsioZMQ
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -stdlib=libc++")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x")
endif ()


find_package(Boost  REQUIRED)
find_package(GLM    REQUIRED)
find_package(Bregex REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS} 
   ${Bregex_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
)

add_definitions( 
   ${Boost_DEFINITIONS} 
   ${Bregex_DEFINITIONS} 
   ${GLM_DEFINITIONS} 
)

add_library( ${name}  SHARED
    NumpyEvt.cpp
    NPYBase.cpp
    NPY.cpp
    ViewNPY.cpp
    MultiViewNPY.cpp
    GLMPrint.cpp
    GLMFormat.cpp
    dirutil.cpp
    stringutil.cpp
    jsonutil.cpp
    timeutil.cpp
    Lookup.cpp
    G4StepNPY.cpp
    TorchStepNPY.cpp
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
    NLog.cpp
    NSphere.cpp
    NPlane.cpp
    NQuad.cpp
    NPart.cpp
    NPrism.cpp
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
    md5digest.cpp
)

set(LIBRARIES
      ${Boost_LIBRARIES}
      ${Bregex_LIBRARIES}
      ${GLM_LIBRARIES}
      -lssl -lcrypto
)

target_link_libraries( ${name} ${LIBRARIES} )

SET(hdrs 
uif.h
ucharfour.h
charfour.h
GLMPrint.hpp
GLMFormat.hpp
NumpyEvt.hpp
NPYBase.hpp
NPY.hpp
ViewNPY.hpp
MultiViewNPY.hpp
G4StepNPY.hpp
TorchStepNPY.hpp
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
NLog.hpp
Parameters.hpp
md5digest.hpp
)

install(TARGETS ${name}  DESTINATION lib)
install(FILES ${hdrs}    DESTINATION include)

add_subdirectory(tests)

