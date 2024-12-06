SysRap : Low Level System Wrapper 
-----------------------------------


**Logging**

.. toctree::

   SLOG


**Serialization**

* NP.hh : workhorse array serialization into NumPy .npy format files 
* NPU.hh : base methods used by NP.hh 
* NPX.h : Higher level serialization utilities 
* NPFold.h : Manager of in-memory recursive "folders" of arrays with persisting to file system folders  


**Image and Font Utilities : Single-file public domain (or MIT licensed) libraries for C/C++**

::

    stb_image.h
    stb_image_write.h
    stb_truetype.h


**Enumerations and Associated**

::

    OpticksCSG.h
    OpticksGenstep.h
    OpticksPhoton.h
             


**Utilities on top of CUDA vector_types vector_functions**

::

    scuda.h


**Instrumenting Photon Propagations**

sctx.h
sphoton.h
stag.h





**Solid geometry headers**

**Structural geometry headers**



**Preparation of placement transforms**

::

    SPlace.h
    SPlaceCircle.h
    SPlaceCylinder.h
    SPlaceRing.h
    SPlaceSphere.h


**Geant4 Related**

::

    S4.h
    S4MaterialPropertyVector.h
    S4RandomArray.h
    S4RandomMonitor.h


**CUDA Thrust based Utilities**

::

    iexpand.h
    strided_range.h


**Pointer based Node Tree Persisting**

::

    s_pool.h
    sn.h
    sndtree.h
    stv.h

**3D view math**


* SPresent.h : present glm vectors and matrix


::

    SBAS.h
    SGLM.h
    SCAM.h
    SGLFW.h

**Random Numbers**

* SRandom.h : pure virtual getFlat methods
* SUniformRand.h
* scurand.h
* s_mock_curand.h

**Config Enumeration, Presentation**

* SRG.h : Raygen mode
* SRM.h : Running mode 

**File system Utilities**

* SDir.h : header only dirent.h directory listing paths with supplied ext 

**Identity Mechnanics**

SName.h
SNameOrder.h


**CUDA Control**

* SCVD.hh : promotes CVD envvar into CUDA_VISIBLE_DEVICES
* SLaunchSequence.h
* salloc.h : debug out of memory errors on device


**Debugging**

SBacktrace.h
SDBG.h :  NONE, BACKTRACE, SUMMARY, CALLER, INTERRUPT
SPhoton_Debug.h


**Time Utilities**

s_time.h
schrono.h
stimer.h
STime.hh
STimes.hh


**Genstep base types**

::

    scarrier.h
    scerenkov.h
    sscint.h
    storch.h
    storchtype.h

**Ascii Rendering**

scanvas.h

**Unclassified**


SBnd.h
SCF.h
SCSGOptiX.h
SCenterExtentFrame.h
SComp.h
SGenerate.h
SIntersect.h

SSimtrace.h
SStackFrame.h
STrackInfo.h
SVolume.h


s_mock_texture.h


saabb.h
sbb.h
sbibit.h
sbit_.h
sbitmask.h
sboundary.h
sc4u.h


sdebug.h
sdigest.h
sdirect.h
sdomain.h
sevent.h
sfactor.h
sflow.h
sfmt.h
sframe.h
sfreq.h
sgeomdefs.h
sgs.h
slog.h
smath.h

snode.h

spa.h
spack.h
spath.h
sphit.h
spho.h

sqat4.h
squad.h
squadlite.h
srec.h


sseq.h
ssincos.h
ssolid.h
sstate.h
sstr.h
ssys.h
st.h

stmm.h

stra.h
stran.h
stree.h
strid.h



sview.h
sxf.h
sxyz.h
tcomplex.h
CheckGeo.hh


OPTICKS_LOG.hh
OpticksPhoton.hh
PLOG.hh
PlainFormatter.hh
SASCII.hh
SAbbrev.hh
SAr.hh
SArgs.hh
SArr.hh
SArrayDigest.hh
SBase36.hh
SBit.hh
SBitSet.hh
SBuf.hh
SCenterExtentGenstep.hh
SColor.hh
SComponent_OLD.hh
SConstant.hh
SCount.hh
SCtrl.hh
SCurandState.hh
SDice.hh
SDigest.hh
SDigestNP.hh
SDirect.hh
SEnabled.hh
SEvent.hh
SEventConfig.hh
SEvt.hh
SFastSimOpticalModel.hh
SFastSim_Debug.hh
SFrameConfig.hh
SFrameGenstep.hh
SGDML.hh
SGenstep.h
SGeo.hh
SGeoConfig.hh
SIMG.hh
SId.hh
SLOG.hh
SLOGF_INIT.hh
SLOG_INIT.hh
SLabelCache.hh
SLauncher.hh
SLogger.hh
SMap.hh
SMath.hh
SMeta.hh
SMockViz.hh
SNameVec.hh
SOpBoundaryProcess.hh
SOpticks.hh
SOpticksEvent.hh
SOpticksKey.hh
SOpticksResource.hh
SPPM.hh
SPack.hh
SPairVec.hh
SPath.hh
SPhiCut.hh
SProc.hh
SProp.hh
SRand.hh
SRenderer.hh
SRng.hh
SRngSpec.hh
SSeq.hh
SSim.hh
SSortKV.hh
SStr.hh
SSys.hh
STTF.hh
SThetaCut.hh

STranche.hh
SU.hh
SVec.hh
SYSRAP_API_EXPORT.hh
SYSRAP_BODY.hh
SYSRAP_HEAD.hh
SYSRAP_LOG.hh
SYSRAP_TAIL.hh
S_freopen_redirect.hh
S_get_option.hh
md5.hh
scsg.hh
snd.hh





