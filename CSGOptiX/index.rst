CSGOptiX : expt with OptiX 7 geometry and rendering 
======================================================

TODO
-----

* arrange default env settings such that the bare executable can run 


Census
-------

=====================  ====================  =================   ============================
 commandline             A:Darwin/OptiX 5      B:Linux/OptiX 6    C:Linux/OptiX 7
=====================  ====================  =================   ============================
CSGOptiXRender            fail 1               fail 2 OR hang      OK : long view, no detail
CSGOptiXSimulate                                                   OK 
./cxr_overview.sh         OK                   fail 1              OK 
./cxr_view.sh             fail 1               hang                OK : PMTs, no struts 
./cxr_solid.sh            fail 1               hang                OK 
./cxr.sh 
=====================  ====================  =================   ============================


A
   build: cx ; om 
B
   build: cx ; om 
   rsync: cx ; ./grab.sh 
C
   build: cx ; ./build7.sh 
   rsync: cx ; ./grab.sh 



CSGOptiXSimulate
-----------------

* requires OPTICKS_KEYDIR envvar (+OPTICKS_KEY?) pointing to a recent geocache with LS_ori material 


scratch workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CSGOptiX::prepareSimulateParam

1. upload gensteps
2. create seeds from the gensteps (QSeed)
3. set gensteps, seeds, photons refs in Params 


4. optix7 launch 
5. download photons 








Failure Modes
----------------

1::

    2021-08-20 10:47:27.933 INFO  [1880522] [CSGOptiX::render@287] [
    2021-08-20 10:47:27.933 INFO  [1880522] [Six::launch@437] [ params.width 1920 params.height 1080
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuMemcpyDtoHAsync( dstHost, srcDevice, byteCount, hStream.get() ) returned (700): Illegal address)
    Abort trap: 6
    epsilon:CSGOptiX blyth$ 


2::

    2021-08-20 19:21:37.525 INFO  [269834] [Six::createContextBuffer@99] node_buffer 0x7f7445a26c00
    terminate called after throwing an instance of 'optix::Exception'
      what():  Invalid value (Details: Function "RTresult _rtBufferSetDevicePointer(RTbuffer, int, void*)" caught exception: Setting buffer device pointers for devices on which OptiX isn't being run is disallowed.)
    Aborted (core dumped)




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

OptiX7Test.cu
    compiled into ptx that gets loaded by PIP to create the GPU pipeline, with OptiX 7 entry points::
    
    __raygen__rg
    __miss__ms
    __intersection__is
    __closesthit__ch 

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

OptiX6Test.cu geo_OptiX6Test.cu
    compiled into ptx that gets loaded by Six to create OptiX < 7 pipeline


 

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



