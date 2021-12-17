CSGOptiX : expt with OptiX 7 geometry and rendering 
======================================================


3D render scripts
------------------

cxr.sh
    script that runs the CSGOptiXRender executable
    this is a basis script that most of the below scripts invoke after setting controlling envvars 

    [not intended to be used bare, this script is invoked from other scripts]

cxr_overview.sh
    overview viewpoint by targeting the  
    uses envvar MOI=-1 to set an overview viewpoint intended to see the entire geometry
    principal controls are EMM envvar for excluding/selecting solids  

    CSGFoundry::parseMOI/CSGName::parseMOI parses MOI string eg sWorld:0:0 into three integers:
    (export CSGName=INFO to see whats happening)

    Midx 
       mesh index (equivalent for Geant4 LV index or Solid index, selecting a shape) 
    mOrd
       mesh ordinal (when more than one such mesh in geometry this is used to select one of them)
    Iidx 
       instance index      

    [tested Dec 2021]

cxr_scan.sh
    runs one of the other scripts (default cxr_overview.sh) multiple times with different values
    for the EMM envvar which is picked up by the basis cxr.sh script and fed to the executable
    via the "-e" option which is short for --enabledmergedmesh which feeds into a bitset 
    queryable by Opticks::isEnabledMergedMesh  

    [tested Dec 2021]

cxr_table.sh
    rst table creation using snap.py 
    the data for the table is collected from .json metadata sidecars to .jpg renders, 
    so this works on laptop after grabbing the rendered .jpg and .json sidecars
    
    [tested Dec 2021]

cxr_solid.sh
    single solid render, for example  ./cxr_solid.sh r0@
    NB here solid is in the CSGSolid sense which corresponds to a GMergedMesh

    The SLA envvar set by the script is passed into the executable via option --solid_label
    which is accessed from Opticks::getSolidLabel within CSGOptiX/tests/CSGOptiXRender.cc
    which uses CSGFoundry::findSolidIdx to select one or more solids based on the solid_label 

    [tested Dec 2021]

cxr_solids.sh
    multiple invokations of cxr_solid.sh for different solids, 
    currently using seq 0 9 for running all JUNO solids::
 
       ./cxr_solid.sh r0@
       ./cxr_solid.sh r1@
       ...

    [tested Dec 2021]

cxr_view.sh
    sets MOI viewpoint (sWaterTube) deep within geometry  
    and invokes ./cxr.sh 

    [tested Dec 2021]

cxr_views.sh
    multiple invokations of cxr_view.sh varying EMM to change included geometry



cxr_flight.sh
    creates sequence of jpg snaps and puts them together into mp4 



cxr_demo.sh
cxr_demos.sh
cxr_demo_find.sh

cxr_geochain.sh

cxr_pub.sh

cxr_rsync.sh


../bin/flight.sh 

   flight-render-jpg  
       uses single OpFlightPathTest executable invokation with --flightconfig option 
       to create potentially many .jpg snaps into --flightoutdir

   flight-make-mp4
       uses ffmpeg to create .mp4 from the .jpg 


../bin/flight7.sh 

    looks to be an update to flight.sh but using the OptiX 7 executable CSGOptiXFlight

    TODO: this is setting CFBASE, that is no longer the way to pick standard geometry 
    TODO: flight7.sh and flight.sh are too similar, merge these 

    
../docs/misc/making_flightpath_raytrace_movies.rst

    OpSnapTest --savegparts   

    using python machinery to inspect geometry : 


2d render scripts
-------------------------

cxs.sh
cxsd.sh


admin scripts
----------------

grab.sh 
    rsync .jpg .png .mp4 .json etc.. outputs from P:/tmp/blyth/opticks/CSGOptiX/ to local (eg laptop)::

        cx 
        ./grab.sh  

sync.sh
    sync PWD code to remote 

cf.sh

pub.sh
     




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



