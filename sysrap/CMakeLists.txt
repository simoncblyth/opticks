cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(name SysRap)
set(desc "System Level Utilities depending")
project(${name} VERSION 0.1.0 LANGUAGES CXX CUDA)
include(OpticksBuildOptions NO_POLICY_SCOPE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  ## write compile_commands.json into build dir

#[=[
SysRap
========

The default RPATH setup from OpticksBuildOptions of
"set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)"
appears to only work if your executables link to a library
which is in the same directory as the library you are creating.
Fortunately thats easy to do by linking with Opticks::OKConf.

Attempting to detect the CUDA version::

    message( STATUS "CMAKE_CUDA_COMPILER_VERSION  : ${CMAKE_CUDA_COMPILER_VERSION}" )

Requires:

* CMake 3.17
* "LANGUAGES CUDA" on the project line
* BUT: it messes up clang compiler detection in cmake/Modules/OpticksCXXFlags.cmake
* so use envvar OPTICKS_CUDA_NVCC_DIALECT detection in cmake/Modules/OpticksCXXFlags.cmake to kludge this


OpticksCSG.py managed as source despite needing manual regeneration after OpticksCSG.h updates
-------------------------------------------------------------------------------------------------

NB generation of OpticksCSG.py is not automated, unlike the generation of other things like status json.
This is because it is expedient for the generated python to be managed as a source file for simplicity.
It would be cleaner to generate json that a future OpticksCSG.py parses.
But as OpticksCSG.h updates are rare, For now just need to remember to generate manually with::

    cd ~/opticks/sysrap
    ../bin/c_enums_to_python.py OpticksCSG.h > OpticksCSG.py


#]=]

find_package(OKConf REQUIRED CONFIG)
find_package(NLJSON REQUIRED MODULE)
find_package(PLog   REQUIRED MODULE)

find_package(CUDAToolkit)   ## for CUDA::cudart


set(Custom4_VERBOSE ON)
find_package(Custom4 CONFIG)

message(STATUS "${name} Custom4_FOUND:${Custom4_FOUND}  " )


set(WITH_SLOG YES)



set(SYSRAP_GENERATED_HEADER  "sconfig.h")
set(SYSRAP_OPTICKS_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
configure_file( ${SYSRAP_GENERATED_HEADER}.in inc/${SYSRAP_GENERATED_HEADER} )
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/inc/${SYSRAP_GENERATED_HEADER}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})



set(SOURCES
    SYSRAP_LOG.cc
    SLOG.cc

    SCurandStateMonolithic.cc
    SSys.cc
    SProf.cc


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
    SLogger.cc
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
    SEnabled.cc
    SRand.cc

    SMockViz.cc
    SRngSpec.cc

    CheckGeo.cc

    SLabelCache.cc

    SMath.cc

    snd.cc
    scsg.cc

    s_csg.cc
    sn.cc
    s_pa.cc
    s_bb.cc
    s_tv.cc


    SEvent.cc
    SFrameGenstep.cc

    SOpticks.cc
    SOpticksResource.cc
    SOpticksKey.cc
    SThetaCut.cc
    SPhiCut.cc

    SCenterExtentGenstep.cc

    OpticksPhoton.cc

    SU.cu

    SEventConfig.cc
    SFrameConfig.cc
    SGeoConfig.cc

    SEvt.cc
    SSim.cc

    SFastSim_Debug.cc
    SFastSimOpticalModel.cc

    SOpBoundaryProcess.cc
)








set(CURANDLITE_HEADERS
    curandlite/curandStatePhilox4_32_10_OpticksLite.h
)


set(HEADERS
    OPTICKS_LOG.hh

    SYSRAP_LOG.hh
    SYSRAP_API_EXPORT.hh
    SYSRAP_HEAD.hh
    SYSRAP_TAIL.hh
    SLOG.hh
    slog.h

    PLOG.hh
    SLOG_INIT.hh

    SLaunchSequence.h


    SCurandSpec.h
    SCurandChunk.h
    scurandref.h
    SCurandState.h


    SCurandStateMonolithic.hh
    sproc.h
    sprof.h
    SProf.hh
    smeta.h

    SBacktrace.h
    SStackFrame.h
    SProfile.h

    ssys.h
    SSys.hh

    sstr.h
    SStr.hh

    schrono.h
    stimer.h
    sstamp.h

    SBnd.h
    SPrd.h

    SPath.hh
    spath.h
    sdirectory.h

    sdevice.h
    scontext.h

    SSeq.hh
    STranche.hh

    SVec.hh
    svec.h

    SNameVec.hh
    SMap.hh
    SCount.hh

    slist.h
    sfreq.h
    snode.h
    stree.h
    suniquename.h

    sstandard.h
    sprop.h
    sproplist.h
    sidxname.h

    snam.h
    smatsur.h


    strid.h
    sfactor.h

    snd.hh
    scsg.hh

    sndtree.h

    s_pool.h
    s_csg.h
    sn.h
    s_pa.h
    s_bb.h
    s_tv.h

    st.h
    stra.h
    sgeomtools.h

    spa.h
    sxf.h
    sbb.h

    sdigest.h
    SDigest.hh



    SDigestNP.hh
    SDirect.hh
    sdirect.h
    SArrayDigest.hh

    SLauncher.hh
    SRenderer.hh
    SRandom.h
    s_seq.h
    S4Random.h

    SCF.h

    SCtrl.hh
    SGeo.hh
    SConstant.hh
    SEvt.hh

    STimes.hh
    SLogger.hh
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
    s_time.h

    SPPM.hh
    sppm.h

    SColor.hh
    SId.hh
    SGDML.hh
    SASCII.hh
    SAbbrev.hh
    SPack.hh
    SBit.hh
    SBitSet.h
    SEnabled.hh
    SRand.hh

    SMockViz.hh
    SRngSpec.hh

    CheckGeo.hh
    SLabelCache.hh

    SRng.hh
    SBuf.hh

    SName.h
    SLabel.h

    smath.h
    sblackbody.h
    tcomplex.h

    srng.h
    sbuild.h
    srngcpu.h
    scurand.h

    s_mock_texture.h
    stexture.h

    njuffa_erfcinvf.h
    s_mock_erfcinvf.h


    squad.h
    squadx.h

    sphoton.h
    sphit.h
    spho.h
    sgs.h
    srec.h
    srec.h
    sseq.h
    sseq_index.h
    sseq_record.h
    sseq_array.h

    sframe.h
    sfr.h
    SCE.h
    SCSGOptiX.h

    SPMT.h
    s_pmt.h
    SPMTAccessor.h

    storchtype.h

    storch.h
    scarrier.h
    scerenkov.h
    sscint.h
    sevent.h
    sstate.h
    sctx.h
    salloc.h

    SGenerate.h

    sflow.h
    sqat4.h
    saabb.h
    stran.h
    stmm.h

    SPlace.h
    SPlaceSphere.h
    SPlaceRing.h
    SPlaceCylinder.h
    SPlaceCircle.h

    ssincos.h
    sxyz.h
    sc4u.h
    stag.h

    sbit_.h
    sbibit.h

    sview.h
    strided_range.h
    iexpand.h


    SOpticks.hh
    sdomain.h

    SDBG.h

    SOpticksResource.hh

    SCMD.h

    SRecord.h
    SGen.h

    SGLM.h
    SGLM_Parse.h
    SGLM_Modifiers.h
    SGLM_Arcball.h

    GL_CHECK.h

    SScene.h
    SMesh.h
    SMeshGroup.h

    SGLFW.h
    SGLFW_Keys.h
    SGLFW_check.h
    SGLFW_Buffer.h
    SGLFW_VAO.h
    SGLFW_Attrib.h
    SGLFW_Program.h
    SGLFW_Mesh.h
    SGLFW_Scene.h
    SGLFW_Record.h
    SGLFW_Gen.h
    SGLFW_Evt.h
    SGLFW_GLEQ.h

    SGLDisplay.h
    SCUDA_OutputBuffer.h
    SGLFW_CUDA.h

    SCUDA_Mesh.h
    SCU_BufferView.h
    SCUDA_MeshGroup.h
    SCSGPrimSpec.h

    scuda.h
    scuda_pointer.h

    SOPTIX.h
    SOPTIX_Context.h
    SOPTIX_Properties.h
    SOPTIX_Options.h
    SOPTIX_getPRD.h
    SOPTIX_Desc.h
    SOPTIX_Accel.h
    SOPTIX_Binding.h
    SOPTIX_MeshGroup.h
    SOPTIX_Module.h
    SOPTIX_OPT.h
    SOPTIX_Pipeline.h
    SOPTIX_SBT.h
    SOPTIX_Scene.h

    SOPTIX_BuildInput.h
    SOPTIX_BuildInput_Mesh.h
    SOPTIX_BuildInput_CPA.h
    SOPTIX_BuildInput_IA.h

    gleq.h

    SDir.h
    SRG.h
    SCAM.h
    SBAS.h

    SEvt.hh

    SOpticksKey.hh
    SMath.hh

    SEvent.hh
    SGenstep.h
    sslice.h

    SFrameGenstep.hh

    SThetaCut.hh
    SPhiCut.hh


    SCenterExtentFrame.h
    SPresent.h

    SCenterExtentGenstep.hh
    scanvas.h

    OpticksGenstep.h
    OpticksPhoton.h
    OpticksPhoton.hh

    CUDA_CHECK.h
    SCU.h
    SCU_.h
    OPTIX_CHECK.h


    SU.hh
    SComponent_OLD.hh
    SComp.h

    SEventConfig.hh
    SRM.h
    SFrameConfig.hh
    SGeoConfig.hh

    np.h
    NP.hh
    NPU.hh
    NPX.h
    NPFold.h
    SSim.hh
    SPropMockup.h

    S4Material.h
    S4MaterialPropertyVector.h

    SNameOrder.h
    S4.h
    S4Surface.h
    S4TreeBorder.h

    S4RandomMonitor.h
    S4RandomArray.h

    SProcessHits_EPH.h

    SFastSim_Debug.hh
    SFastSimOpticalModel.hh
    STrackInfo.h
    SVolume.h
    SPhoton_Debug.h
    SUniformRand.h

    sgeomdefs.h
    sfmt.h
    ssolid.h
    SIntersect.h
    SSimtrace.h

    SOpBoundaryProcess.hh
)

set(RETIRED_SOURCES
    SProc.cc
)

set(RETIRED_HEADERS
    SProc.hh
)










# using absolute ANADIR avoids build environment requiring it in PATH
get_filename_component(ANADIR ../ana ABSOLUTE)
get_filename_component(BINDIR ../bin ABSOLUTE)

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

#[=[

   cd ~/opticks/sysrap ; ../ana/enum_.py OpticksGenstep.h --quiet --simple --inipath /tmp/OpticksGenstep_Enum.ini && cat /tmp/OpticksGenstep_Enum.ini
#]=]


## HMM:the flags change very infrequently : easier to code generate into source directories for discoverability
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
    list(APPEND HEADERS   stb_image.h  stb_image_write.h)
    list(APPEND HEADERS   SIMG.h  )
    list(APPEND HEADERS   SIMG_Frame.h  )
endif()


set(WITH_STTF YES)
if(WITH_STTF)
    list(APPEND HEADERS stb_truetype.h)
    list(APPEND HEADERS STTF.h)
endif()


set(WITH_NP YES)


add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )


string (REPLACE "," ";" COMPUTE_ARCHITECTURES_STR "${COMPUTE_ARCHITECTURES}")
message( STATUS "COMPUTE_ARCHITECTURES     : ${COMPUTE_ARCHITECTURES}" )
message( STATUS "COMPUTE_ARCHITECTURES_STR : ${COMPUTE_ARCHITECTURES_STR}" )
set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES "${COMPUTE_ARCHITECTURES_STR}")


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
target_link_libraries(${name}  Opticks::NLJSON)
endif()


if(UNIX AND NOT APPLE)
message(STATUS "adding ssl crypto for UNIX AND NOT APPLE")
target_link_libraries( ${name} ssl crypto )
endif()

set(SysRap_VERBOSE OFF)
if(SysRap_VERBOSE)
  echo_target_std(Opticks::OKConf)
  #echo_target_std(${name})
endif()


target_include_directories( ${name} PUBLIC
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
   $<BUILD_INTERFACE:${OPTICKS_PREFIX}/externals/glm/glm>
   $<INSTALL_INTERFACE:externals/glm/glm>
)



#[=[
SysRap Compile Definitions
----------------------------

former  WITH..SND
   switched the below to use the old inflexible snd.hh CSG node impl,
   instead of the default more flexible sn.h impl:

   sysrap/stree.h
       WITH..SND saves CSG nodes in CSGFoundry/SSim/stree/csg
       NOT:WITH..SND saves CSG nodes in CSGFoundry/SSim/stree/_csg

   u4/U4Tree.h u4/U4Solid.h u4/U4Polycone.h
       WITH..SND uses snd.hh idx int as CSG root
       NOT:WITH..SND uses sn.h sn* pointer as CSG root

WITH_CHILD
   sysrap/sn.h uses child vector of sn pointers supporting n-ary trees

NOT:WITH_CHILD
   sysrap/sn.h uses left, right sn pointers supporting only binary trees

DEBUG_TAG

   $<$<CONFIG:Debug>:DEBUG_TAG>

   ONLY LEAVE ENABLED WHEN DOING RANDOM ALIGNED RUNNING TESTS

   needed for random aligned running in multiple pkgs: sysrap, qudarap, u4
   however making this PUBLIC makes rebuilding real heavy
   so must rely on making coordinated switches when doing random aligned running



PLOG_LOCAL
   changes visibility of plog external symbols, allowing better
   integration with packages (like junosw) that do not hide
   symbols by default

RNG_XORWOW/RNG_PHILOX/RNG_PHILITEOX
   Use one of these to pick the curandState implementation, in srng.h
   picking RNG_XORWOW requires curandState files which slows
   down initialization greatly when simulating many millions
   of photons

#]=]


if(Custom4_FOUND)
   target_compile_definitions( ${name} PUBLIC WITH_CUSTOM4 )
   target_include_directories( ${name} PUBLIC ${Custom4_INCLUDE_DIR})
endif()


set(SYSRAP_COMPILE_DEFINITIONS
      $<$<CONFIG:Debug>:CONFIG_Debug>
      $<$<CONFIG:RelWithDebInfo>:CONFIG_RelWithDebInfo>
      $<$<CONFIG:Release>:CONFIG_Release>
      $<$<CONFIG:MinSizeRel>:CONFIG_MinSizeRel>

      OPTICKS_SYSRAP
      WITH_CHILD
      PLOG_LOCAL
      RNG_PHILOX

      $<$<CONFIG:Debug>:DEBUG_PIDX>
      $<$<CONFIG:Debug>:DEBUG_PIDXYZ>
      $<$<CONFIG:Release>:PRODUCTION>
)


target_compile_definitions( ${name}
    PUBLIC
           ${SYSRAP_COMPILE_DEFINITIONS}
)



if(WITH_STTF)
target_compile_definitions( ${name} PUBLIC WITH_STTF)
endif()
if(WITH_SLOG)
target_compile_definitions( ${name} PUBLIC WITH_SLOG)
endif()







set(OPTIX_SOURCES)
list(APPEND OPTIX_SOURCES
    SOPTIX.cu
)

set(OpticksOptiX_VERBOSE ON)
find_package(OpticksOptiX REQUIRED MODULE)

message(STATUS "${name} OPTIX_SOURCES        : ${OPTIX_SOURCES} ")
message(STATUS "${name} OpticksOptiX_VERSION : ${OpticksOptiX_VERSION} ")
message(STATUS "${name} OpticksOptiX_INCLUDE : ${OpticksOptiX_INCLUDE} ")
message(STATUS "${name} COMPUTE_CAPABILITY   : ${COMPUTE_CAPABILITY} ")

add_library( ${name}_OPTIX OBJECT ${OPTIX_SOURCES} )
add_dependencies( ${name} ${name}_OPTIX )


set_target_properties( ${name}_OPTIX
    PROPERTIES
         CUDA_PTX_COMPILATION ON
         POSITION_INDEPENDENT_CODE ON
         CUDA_SEPARABLE_COMPILATION ON
         CUDA_ARCHITECTURES "${COMPUTE_CAPABILITY}"    ## single integer CC of oldest GPU
)

target_include_directories(${name}_OPTIX
   PUBLIC
      ${OpticksOptiX_INCLUDE}
      $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_compile_options( ${name}_OPTIX
   PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O2 --use_fast_math -Xcompiler -fPIC>
)

install( TARGETS ${name}_OPTIX OBJECTS DESTINATION optix )








# <findname>_targets is my convention
message(STATUS "PLog_targets:${PLog_targets}")

## top_meta_target(TOPMETA..)  function call from cmake/Modules/TopMetaTarget.cmake creates TOPMETA string into this scope
top_meta_target(PLog_TOPMETA    "PLog" "${PLog_targets}" )

message(STATUS "${PLog_TOPMETA}")

set(PC_VERBOSE ON)
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL TOPMATTER  "
## SysRap TOPMATTER

${PLog_TOPMETA}

## trying to pass compile definitions this way difficult

## end SysRap TOPMATTER
")

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


set( OLD_SCRIPTS
    __init__.py
)

set( PY_SCRIPTS
    __init__.py
    sframe.py
    sevt.py
    smath.py
    OpticksCSG.py
)

set( SH_SCRIPTS
    SABTest.sh
    SABTest.py
)

install(PROGRAMS ${PY_SCRIPTS}  DESTINATION py/opticks/sysrap)
install(PROGRAMS ${SH_SCRIPTS}  DESTINATION bin)

install(FILES ${CURANDLITE_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/curandlite)
install(DIRECTORY gl/ DESTINATION gl)

add_subdirectory(tests)



