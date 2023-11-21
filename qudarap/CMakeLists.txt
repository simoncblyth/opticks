cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name QUDARap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions NO_POLICY_SCOPE)      


set(FLAGS_VERBOSE ON)

set(LIBS  )


find_package(SysRap REQUIRED CONFIG)
set(LIBS ${LIBS} Opticks::SysRap)  


set(Custom4_VERBOSE ON)
find_package(Custom4 CONFIG)

message(STATUS "${name} Custom4_FOUND:${Custom4_FOUND}  " )

#[=[
A successful find of Custom4 runs some CMake config, see::

    epsilon:qudarap blyth$ cat ${OPTICKS_PREFIX}_externals/custom4/0.1.9/lib/Custom4-0.1.9/Custom4Config.cmake | grep WITH
    set(Custom4_DEFINITIONS -DWITH_CUSTOM4 )
          INTERFACE_COMPILE_DEFINITIONS "WITH_CUSTOM4"

#]=]




set(OpticksCUDA_VERBOSE ON)
find_package(OpticksCUDA REQUIRED MODULE)


set(SOURCES
    QUDARAP_LOG.cc

    QCurandState.cc
    QCurandState.cu

    QSim.cc
    QSim.cu

    QPMT.cc
    QPMT.cu 

    QEvent.cc
    QEvent.cu

    QDebug.cc
    QState.cc

    QU.cc

    QRng.cc  
    QRng.cu

    QTex.cc
    QTexRotate.cc
    QTex.cu
    QTexMaker.cc
    QTexLookup.cc
    QTexLookup.cu

    QBase.cc
    QBnd.cc
    QBnd.cu


    QOptical.cc
    QOptical.cu 

    QScint.cc
    QScint.cu

    QCerenkovIntegral.cc
    QCerenkov.cc
    QCerenkov.cu 

    QCK.cc
    QCK.hh

    QPoly.cc
    QPoly.cu

    QProp.cc
    QProp.cu

    QMultiFilm.cc
    QMultiFilm.cu

    QBuf.cc
)

SET(HEADERS
    QUDARAP_LOG.hh
    QUDARAP_API_EXPORT.hh
    QUDA_CHECK.h


    QCurandState.hh
    qcurandstate.h


    QSim.hh
    qsim.h

    QPMT.hh
    qpmt.h

    qgs.h
    qprop.h
 
    QEvent.hh

    QDebug.hh
    qdebug.h

    QU.hh
    QRng.hh

    QTex.hh
    QTexRotate.hh
    QTexMaker.hh
    QTexLookup.hh

    QState.hh

    qmultifilm.h 

    QBase.hh
    qbase.h 

    QBnd.hh
    qbnd.h

    QOptical.hh

    QScint.hh
    qscint.h

    QCerenkovIntegral.hh
    QCerenkov.hh
    qcerenkov.h

    QPoly.hh

    QProp.hh
    
    QMultiFilm.hh

    QBuf.hh
)


# OLD_WAY needed GGeo
# Opticks::SysRap
# Opticks::GGeo

if(UNIX AND NOT APPLE)
  set(LIBS ${LIBS} ssl )  
endif()








CUDA_ADD_LIBRARY( ${name} ${SOURCES} )

get_target_property(ILL ${name} INTERFACE_LINK_LIBRARIES)
message(STATUS "${name} INTERFACE_LINK_LIBRARIES:${ILL} " )

#[=[
As QUDARap only requires a single header-only header from Custom4,
not the library it is unnecessarily adding dependency on the lib and 
all the Geant4 libs when hooking up the full target with::

   # set(LIBS ${LIBS} Custom4::Custom4) 

So instead just hookup the include_directories and 
compile_definitions not the full target. 
#]=]

if(Custom4_FOUND)
   target_compile_definitions( ${name} PUBLIC WITH_CUSTOM4 )
   target_include_directories( ${name} PUBLIC ${Custom4_INCLUDE_DIR})
endif()

target_link_libraries( ${name} ${LIBS})
target_compile_definitions( ${name} PUBLIC OPTICKS_QUDARAP )

#[=[

https://stackoverflow.com/questions/34490294/what-does-configdebugrelease-mean-in-cmake

   $<CONFIG:Debug>   1 for Debug, 0 otherwise
   $<1:X>            X 
   $<0:X>            Empty 

#]=]


#target_compile_definitions( ${name} PUBLIC DEBUG_PIDX )
#target_compile_definitions( ${name} PUBLIC DEBUG_TAG )


target_compile_definitions( ${name}
    PUBLIC
      OPTICKS_QUDARAP 
      $<$<CONFIG:Debug>:DEBUG_TAG>
      $<$<CONFIG:Debug>:DEBUG_PIDX>
)


#[=[

WITH_THRUST compile definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* WITH_THRUST is needed for the Custom4 GPU complex math functions used by qpmt.h 

* However Custom4 also used on CPU without WITH_THRUST (eg by junosw), so 
  the WITH_THRUST definition must be PRIVATE to avoid interference with other
  Custom4 usage

#]=]

target_compile_definitions( ${name} PRIVATE WITH_THRUST )

#target_compile_definitions( ${name} PRIVATE DEBUG_TIME )
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )

# see notes/issues/QUDARap_symbol_visibility_warnings.rst 
#set_target_properties( ${name} PROPERTIES CXX_VISIBILITY_PRESET hidden )



install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)


