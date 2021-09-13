cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name QUDARap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)      

if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()


set(FLAGS_VERBOSE ON)

#find_package(SysRap REQUIRED CONFIG)
find_package(OpticksCore REQUIRED CONFIG) # needed for OpticksEvent OpticksGenstep enum 
#find_package(GGeo REQUIRED CONFIG)       # OLD_WAY needed GGeo, but thats too high level for comfort

set(OpticksCUDA_VERBOSE ON)
find_package(OpticksCUDA REQUIRED MODULE)


set(SOURCES
    QUDARAP_LOG.cc

    QSim.cc
    QSim.cu

    QEvent.cc
    QEvent.cu

    QU.cc

    QRng.cc  
    QRng.cu

    QTex.cc
    QTex.cu

    QBnd.cc
    QBnd.cu

    QScint.cc
    QScint.cu

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

    QSim.hh
    qsim.h

    qgs.h
    qprop.h
    qcurand.h
 
    QEvent.hh
    qevent.h

    QU.hh
    QRng.hh

    QTex.hh
    QBnd.hh
    QScint.hh
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


