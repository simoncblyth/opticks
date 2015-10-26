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
find_package(Bregex REQUIRED)
find_package(GLM    REQUIRED)

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
    Index.cpp
    Timer.cpp
    Times.cpp
    NCache.cpp
    NSensor.cpp
    NSensorList.cpp
    NSlice.cpp
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
stringutil.hpp
jsonutil.hpp
timeutil.hpp
Lookup.hpp
Types.hpp
Index.hpp
Timer.hpp
Times.hpp
NCache.hpp
NSensor.hpp
NSensorList.hpp
NSlice.hpp
Parameters.hpp
md5digest.hpp
)

install(TARGETS ${name}  DESTINATION lib)
install(FILES ${hdrs}    DESTINATION include)

add_subdirectory(tests)

