cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name SysRap)
set(desc "System Level Utilities depending")
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions NO_POLICY_SCOPE)


#[=[
SysRap
========

The default RPATH setup from OpticksBuildOptions of 
"set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)"
appears to only work if your executables link to a library 
which is in the same directory as the library you are creating.  
Fortunately thats easy to do by linking with Opticks::OKConf.

#]=]

find_package(OKConf REQUIRED CONFIG)
find_package(NLJSON REQUIRED MODULE)
find_package(PLog   REQUIRED MODULE)
#find_package(CUDA REQUIRED)

find_package(OpticksCUDA REQUIRED MODULE)


include_directories(
    ${OPTICKS_PREFIX}/externals/glm/glm
)


set(WITH_PLOG YES)

set(SOURCES
    SYSRAP_LOG.cc 
    PLOG.cc 
    SBacktrace.cc
    SFrame.cc
    SProc.cc
    SSys.cc
    SStr.cc
    SPath.cc
    SSeq.cc
    STranche.cc
    SAr.cc
    SArr.cc
    SArgs.cc
    SPPM.cc
    SVec.cc
    SNameVec.cc
    SMap.cc
    SCount.cc
    SDigest.cc
    SDigestNP.cc
    SDirect.cc
    SArrayDigest.cc

    SLauncher.cc
    SRenderer.cc
    SCtrl.cc
    SGeo.cc
    SConstant.cc

    STimes.cc
    SLog.cc
    SBase36.cc
    SSortKV.cc
    SPairVec.cc
    md5.cc
    SPPM.cc
    SColor.cc
    SId.cc
    SGDML.cc
    S_get_option.cc
    STime.cc
    SASCII.cc
    SAbbrev.cc
    SPack.cc
    SBit.cc
    SBitSet.cc
    SEnabled.cc
    SRand.cc

    SMockViz.cc
    SRngSpec.cc

    CheckGeo.cc

    SGLM.cc
    SLabelCache.cc

    SMath.cc
    SEvent.cc
    SOpticks.cc
    SOpticksResource.cc
    SOpticksKey.cc
    SThetaCut.cc 
    SPhiCut.cc 

    SCenterExtentFrame.cc
    SCenterExtentGenstep.cc

    OpticksPhoton.cc

    SU.cu 

    SEventConfig.cc
) 
    


set(HEADERS
    OPTICKS_LOG.hh

    SYSRAP_LOG.hh
    SYSRAP_API_EXPORT.hh
    SYSRAP_HEAD.hh
    SYSRAP_TAIL.hh
    PLOG.hh
    PLOG_INIT.hh  

    SProc.hh
    SBacktrace.hh
    SFrame.hh
    SSys.hh
    SStr.hh
    SPath.hh
    SSeq.hh
    STranche.hh
    SVec.hh
    SNameVec.hh
    SMap.hh
    SCount.hh
    SDigest.hh
    SDigestNP.hh
    SDirect.hh
    SArrayDigest.hh

    SLauncher.hh
    SRenderer.hh
    SCtrl.hh
    SGeo.hh
    SConstant.hh

    STimes.hh
    SLog.hh
    SBase36.hh
    SSortKV.hh
    SPairVec.hh
    OpticksCSG.h
    PlainFormatter.hh
    SAr.hh
    SArr.hh
    SArgs.hh
    S_freopen_redirect.hh
    S_get_option.hh
    STime.hh
    SPPM.hh
    SColor.hh
    SId.hh
    SGDML.hh
    SASCII.hh
    SAbbrev.hh
    SPack.hh
    SBit.hh
    SBitSet.hh
    SEnabled.hh
    SRand.hh

    SMockViz.hh
    SRngSpec.hh

    CheckGeo.hh
    SGLM.hh
    SLabelCache.hh

    SRng.hh
    SBuf.hh

    scuda.h
    scurand.h  
    smath.h 
    s_mock_curand.h
    squad.h

    storch.h
    storchtype.h
    
    sflow.h
    sqat4.h
    saabb.h
    stran.h
    ssincos.h
    sevent.h
    sc4u.h
    sbit_.h
    sbibit.h

    sview.h
    strided_range.h 
    iexpand.h 


    SOpticks.hh
    SOpticksResource.hh
    SOpticksKey.hh
    SMath.hh
    SEvent.hh
    SThetaCut.hh
    SPhiCut.hh


    SCenterExtentFrame.hh
    SCenterExtentGenstep.hh 
    SPresent.hh
    SCanvas.hh

    OpticksGenstep.h
    OpticksPhoton.h
    OpticksPhoton.hh
    SU.hh
    SEvt.hh 

    SEventConfig.hh
)






# using absolute ANADIR avoids build environment requiring it in PATH 
get_filename_component(ANADIR ../ana ABSOLUTE)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR : ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "ANADIR : ${ANADIR}")



set(HGS ${CMAKE_CURRENT_SOURCE_DIR}/OpticksGenstep.h)
set(IGS ${CMAKE_CURRENT_BINARY_DIR}/OpticksGenstep_Enum.ini)
add_custom_command(
   OUTPUT 
         ${IGS}
   COMMAND 
          ${ANADIR}/enum_.py ${HGS} --quiet --simple --inipath ${IGS}
   DEPENDS 
         ${HGS}
)
add_custom_target(PythonGS ALL DEPENDS ${IGS} )




set(HPH ${CMAKE_CURRENT_SOURCE_DIR}/OpticksPhoton.h)
set(IPH ${CMAKE_CURRENT_BINARY_DIR}/OpticksPhoton_Enum.ini)
add_custom_command(
   OUTPUT 
         ${IPH}
   COMMAND 
          ${ANADIR}/enum_.py ${HPH} --quiet --inipath ${IPH}
   DEPENDS 
         ${HPH}
)
add_custom_target(PythonPH ALL DEPENDS ${IPH} )
# ALL : Indicates that this custom target should be added to the 
# default build target so that it will be run every time.



set(CCFL ${CMAKE_CURRENT_SOURCE_DIR}/OpticksPhoton.hh)
set(PYJS ${CMAKE_CURRENT_BINARY_DIR}/OpticksPhoton_Abbrev.json)

add_custom_command(
   OUTPUT 
         ${PYJS}
   COMMAND 
         ${ANADIR}/OpticksPhoton.py ${CCFL} --quiet --jsonpath ${PYJS}
   DEPENDS 
         ${CCFL}
)
add_custom_target(PythonJSON ALL DEPENDS ${PYJS} )


list(APPEND HEADERS 
  ${IGS} 
  ${IPH} 
  ${PYJS}
) 


if(NLJSON_FOUND)  
   message(STATUS "SYSRAP.NLJSON_FOUND")
   list(APPEND SOURCES  SMeta.cc)
   list(APPEND HEADERS  SMeta.hh)
endif()



set(WITH_SIMG YES)
if(WITH_SIMG)
    list(APPEND HEADERS   SIMG.hh  )
    list(APPEND HEADERS   stb_image.h  stb_image_write.h)   
    # seems cannot avoid passing around the stb headers ?
endif()


set(WITH_STTF YES)
if(WITH_STTF)
    list(APPEND HEADERS STTF.hh stb_truetype.h)
endif()


set(WITH_NP YES)
if(WITH_NP)
    list(APPEND HEADERS   NP.hh  NPU.hh  )
endif()


CUDA_ADD_LIBRARY( ${name} ${SOURCES} )


#add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )
#set_target_properties( ${name} 
#                       PROPERTIES 
#                         INTERFACE_DESCRIPTION ${desc}
#                         INTERFACE_PKG_CONFIG_REQUIRES ${preq}
#                     )

set_property( TARGET ${name} PROPERTY INTERFACE_DESCRIPTION ${desc} )
#set_property( TARGET ${name} PROPERTY INTERFACE_PKG_CONFIG_REQUIRES ${preq} ) 


#target_link_libraries( ${name} PUBLIC Opticks::PLog Opticks::OKConf )
target_link_libraries( ${name} Opticks::PLog Opticks::OKConf )

#Opticks::OpticksCUDA 

if(NLJSON_FOUND)
#target_link_libraries(${name} PUBLIC Opticks::NLJSON)
target_link_libraries(${name}  Opticks::NLJSON)
endif()


if(UNIX AND NOT APPLE)
message(STATUS "adding ssl crypto for UNIX AND NOT APPLE")
#target_link_libraries( ${name} PUBLIC ssl crypto )
target_link_libraries( ${name} ssl crypto )
endif()

set(SysRap_VERBOSE OFF)
if(SysRap_VERBOSE)
  echo_target_std(Opticks::OKConf) 
  #echo_target_std(${name}) 
endif()

target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
   ${CUDA_INCLUDE_DIRS}
   ${OPTICKS_PREFIX}/externals/glm/glm
)
target_compile_definitions( ${name} PUBLIC OPTICKS_SYSRAP )

if(WITH_STTF)
target_compile_definitions( ${name} PUBLIC WITH_STTF)
endif()
if(WITH_PLOG)
target_compile_definitions( ${name} PUBLIC WITH_PLOG)
endif()



# <findname>_targets is my convention
message(STATUS "PLog_targets:${PLog_targets}")

## top_meta_target(TOPMETA..)  function call from cmake/Modules/TopMetaTarget.cmake creates TOPMETA string into this scope
top_meta_target(PLog_TOPMETA    "PLog" "${PLog_targets}" )

message(STATUS "${PLog_TOPMETA}")
 
set(PC_VERBOSE ON)
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL TOPMATTER  "
## SysRap TOPMATTER

${PLog_TOPMETA}

## end SysRap TOPMATTER
")

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


set( SCRIPTS 
    __init__.py
    OpticksCSG.py
)
install(PROGRAMS ${SCRIPTS}  DESTINATION py/opticks/sysrap)


add_subdirectory(tests)



