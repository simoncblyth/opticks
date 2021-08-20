CSGOptiX : expt with OptiX 7 geometry and rendering 
======================================================

TODO
-----

* arrange default env settings such that the bare executable can run 


Census
-------


=====================  ====================  =================
 commandline             Darwin/OptiX 5        Linux/OptiX 7      
=====================  ====================  =================
CSGOptiXRender            fail 1 
./cxr_overview.sh         OK
./cxr_view.sh 
./cxr_solid.sh            fail 1  
=====================  ====================  =================



Failure Modes
----------------

Fail 1::

    2021-08-20 10:47:27.933 INFO  [1880522] [CSGOptiX::render@287] [
    2021-08-20 10:47:27.933 INFO  [1880522] [Six::launch@437] [ params.width 1920 params.height 1080
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuMemcpyDtoHAsync( dstHost, srcDevice, byteCount, hStream.get() ) returned (700): Illegal address)
    Abort trap: 6
    epsilon:CSGOptiX blyth$ 



code
-------

tests/CSGOptiXRender.cc
    main that loads and uploads CSGFoundry geometry and creates 
    one or more renders and saves them to jpg   

CSGOptiX.h
    top level struct using either OptiX pre-7 OR 7 

Params.h
    workhorse for communicating between CPU and GPU 

Frame.h
    render pixels holder  

BI.h
    wrapper for OptixBuildInput 
AS.h
    common acceleration structure base struct for GAS and IAS
GAS.h
    bis vector of BI build inputs 
IAS.h
    vector of transforms and d_instances 

GAS_Builder.h
    building OptiX geometry acceleration structure 

IAS_Builder.h
    building OptiX instance acceleration structure 

Binding.h
    GPU/CPU types, including SbtRecord : RaygenData, MissData, HitGroupData (effectively Prim)

PIP.h
    OptiX render pipeline creation from ptx file

SBT.h
    brings together OptiX 7 geometry and render pipeline programs, nexus of control  

Ctx.h
    holder of OptixDeviceContext and Params and Properties instances

Properties.h
    holder of information gleaned from OptiX 7

InstanceId.h
    encode/decode identity info

OPTIX_CHECK.h
    error check macro for optix 7 calls

Six.h
    optix pre-7 rendering of CSGFoundary geometry


 

scripts
---------

build.sh
build7.sh
cf.sh
cxr.sh
cxr_demo.sh
cxr_demo_find.sh
cxr_demos.sh
cxr_flight.sh
cxr_overview.sh
cxr_rsync.sh
cxr_scan.sh


cxr_solid.sh
    single solid render
cxr_solids.sh
    multiple invokations of cxr_solid.sh for different solids
cxr_table.sh
    rst table creation using snap.py 
cxr_view.sh
    sets envvars and invoked ./cxr.sh 
cxr_views.sh
    multiple invokations of cxr_view.sh varying EMM to change included geometry

run.sh 
    invoke cxr_overview.sh 
go.sh
    invoke build.sh and run.sh 
grab.sh 
    rsync outputs from P:/tmp/blyth/opticks/CSGOptiX/ to local 
sync.sh
    sync PWD code to remote 



