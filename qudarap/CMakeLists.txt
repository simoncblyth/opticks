cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name QUDARap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions NO_POLICY_SCOPE)      


set(FLAGS_VERBOSE ON)

find_package(OpticksCore REQUIRED CONFIG) # needed for OpticksEvent OpticksGenstep enum 

set(OpticksCUDA_VERBOSE ON)
find_package(OpticksCUDA REQUIRED MODULE)


set(SOURCES
    QUDARAP_LOG.cc

    QSim.cc
    QSim.cu

    QEvent.cc
    QEvent.cu

    QDebug.cc

    QU.cc

    QRng.cc  
    QRng.cu

    QTex.cc
    QTexRotate.cc
    QTex.cu
    QTexMaker.cc
    QTexLookup.cc
    QTexLookup.cu

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

    QSeed.cc
    QSeed.cu

    QBuf.cc
)

SET(HEADERS
    QUDARAP_LOG.hh
    QUDARAP_API_EXPORT.hh
    QUDA_CHECK.h

    QSim.hh
    qsim.h

    qgs.h
    qprop.h
    qcurand.h
 
    QEvent.hh
    qevent.h

    QDebug.hh
    qdebug.h

    QU.hh
    QRng.hh

    QTex.hh
    QTexRotate.hh
    QTexMaker.hh
    QTexLookup.hh

    qstate.h

    qbnd.h
    QBnd.hh
    QOptical.hh

    QScint.hh
    QCerenkovIntegral.hh
    QCerenkov.hh


    QPoly.hh

    QProp.hh

    QSeed.hh
    QBuf.hh
)

set(LIBS
    Opticks::OpticksCore
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

target_link_libraries( ${name} ${LIBS})
#target_compile_definitions( ${name} PUBLIC OPTICKS_QUDARAP FLIP_RANDOM )
target_compile_definitions( ${name} PUBLIC OPTICKS_QUDARAP )
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )

# see notes/issues/QUDARap_symbol_visibility_warnings.rst 
#set_target_properties( ${name} PROPERTIES CXX_VISIBILITY_PRESET hidden )



install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)


