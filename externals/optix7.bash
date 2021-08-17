##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

optix7-source(){   echo ${BASH_SOURCE} ; }
optix7-vi(){       vi $(optix7-source) ; }
optix7-env(){      olocal- ; }
optix7-sbt(){      open ~/opticks_refs/sbt-s21888-rtx-accelerated-raytracing-with-optix-7.pdf ; }
optix7-usage(){ cat << \EOU

OptiX 7 : Brand New Lower Level API
======================================= 

See Also 
---------

* optix7c- course from Ingo Wald
* env-;rcs- from Vishal Mehta, compute usage of OptiX7 
* owl-;owl-vi  Higher Level Layer on top of OptiX7 including 


* https://github.com/nvpro-samples/optix_advanced_samples/tree/master/src/optixIntroduction


Multiple pipelines, SBT, ...
--------------------------------

* https://forums.developer.nvidia.com/t/multiple-pipelines-and-shared-instancing/170735/2

* https://forums.developer.nvidia.com/t/multiple-raygen-functions-within-same-pipeline-in-optix-7/122305

* https://forums.developer.nvidia.com/t/how-to-handle-multiple-ray-generators/83446



From OptiX 7.2 Release Notes : **Specialization**
-----------------------------------------------------

What's New in 7.2.0 

Specialization is a powerful new feature that allows
renderers to maintain generality while increasing performance on specific use
cases. A single version of the PTX can be supplied to OptiX and specialized to
toggle specific features on and off. The OptiX compiler is leveraged to fold
constant values and elide complex code that is not required by a particular
scene setup. Specialized values are supplied during module creation with
OptixModuleCompileOptions::boundValues. See the Programming Guide section
6.3.1, “Parameter specialization”, and the optixBoundValues sample.

* https://raytracing-docs.nvidia.com/optix7/guide/index.html#program_pipeline_creation#parameter-specialization


Driver Versions for each OptiX Release
------------------------------------------

OptiX 7.2.0 requires that you install a r455+ driver

OptiX 7.1.0 requires that you install a r450+ driver.

OptiX 7.0.0 requires that you install the 435.80 driver on Windows or the 435.12 Driver for linux.. 

OptiX 6.5.0 requires that you install the 436.02 driver on Windows or the 435.17 Driver for linux.


From Downloads page https://developer.nvidia.com/designworks/optix/download
------------------------------------------------------------------------------

7.2.0 : Requires NVIDIA R456.71 driver or newer for Windows and 455.28 or newer for Linux..

6.5.0 : NOTE: Requires NVIDIA R435.80 driver or newer. You may need a Beta Driver for certain operating systems.



Current Driver Versions on different machines
------------------------------------------------


=================  =======================  =====================  ===================
Machine              GPUs                     Driver                CUDA 
=================  =======================  =====================  ===================
Precision Gold       TITAN V, TITAN RTX       435.21                10.1 
New                  Quadro RTX 8000          460.56                11.2
Cluster              Tesla V100-SXM2          450.36.06             11.0
=================  =======================  =====================  ===================



::

    blyth@localhost ~]$ nvidia-smi
    Wed Mar 17 17:40:00 2021       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.56       Driver Version: 460.56       CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Quadro RTX 8000     Off  | 00000000:73:00.0 Off |                  Off |
    | 33%   30C    P8     9W / 260W |    300MiB / 48592MiB |      2%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+


    [blyth@localhost ~]$ df -h 
    Filesystem               Size  Used Avail Use% Mounted on
    devtmpfs                  63G     0   63G   0% /dev
    tmpfs                     63G  111M   63G   1% /dev/shm
    tmpfs                     63G   12M   63G   1% /run
    tmpfs                     63G     0   63G   0% /sys/fs/cgroup
    /dev/mapper/centos-root  944G   17G  928G   2% /
    /dev/nvme0n1p2           5.0G  161M  4.9G   4% /boot
    /dev/nvme0n1p1           5.0G   12M  5.0G   1% /boot/efi
    tmpfs                     13G   28K   13G   1% /run/user/1000
    cvmfs2                   4.0G  699M  3.3G  18% /cvmfs/juno.ihep.ac.cn
    /dev/sda                 7.3T   93M  6.9T   1% /data
    tmpfs                     13G  8.0K   13G   1% /run/user/1001
    tmpfs                     13G     0   13G   0% /run/user/1003
    tmpfs                     13G     0   13G   0% /run/user/1002
    [blyth@localhost ~]$ 



::

    L7[blyth@lxslc714 ~]$ sr
    job-head
    gpu012.ihep.ac.cn
    Thu Mar 18 03:55:32 2021       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 450.36.06    Driver Version: 450.36.06    CUDA Version: 11.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla V100-SXM2...  On   | 00000000:B6:00.0 Off |                    0 |
    | N/A   37C    P0    45W / 300W |      0MiB / 32510MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   

Forum Links
--------------

* https://forums.developer.nvidia.com/t/multiple-pipelines-and-shared-instancing/170735/2
* https://forums.developer.nvidia.com/t/optix-7-breaking-changes/156801/2
* https://forums.developer.nvidia.com/t/best-way-to-turn-entities-on-off-during-ray-tracing-in-optix/165436/5




GPU Mental Model : Chapter 33. Implementing Efficient Parallel Data Structures on GPUs
---------------------------------------------------------------------------------------------

* https://developer.nvidia.com/gpugems/gpugems2/part-iv-general-purpose-computation-gpus-primer/chapter-33-implementing-efficient


Property Access
-----------------

* https://github.com/NVIDIA/OptiX_Apps/tree/master/apps
* https://github.com/NVIDIA/OptiX_Apps/blob/master/apps/rtigo3/src/Device.cpp


DirectX raytracing is using the same SBT
-------------------------------------------

* ~/opticks_refs/unofficial_RayTracingGems_v1.7.pdf

* "Ray Tracing Gems" v1.7 Chapter 3 "Introduction to DirectX raytracing"  
* although describing DirectX the underlying SBT is the same 
* actually the VulkanRT, DirectX RT and OptiX API all very similar wrt the SBT
* https://www.realtimerendering.com/raytracinggems/
* http://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.7.pdf


GTC ON DEMAND and other learning links
-----------------------------------------

* https://www.nvidia.com/en-us/gtc/on-demand/?search=OptiX

* https://developer.nvidia.com/blog/how-to-get-started-with-optix-7/


:google:`optix sbt`
-----------------------

* https://www.willusher.io/graphics/2019/11/20/the-sbt-three-ways

Thus, there should be at least one Shader Record in the table for each unique
combination of shader functions and embedded parameters.
It is possible to write the same shader record multiple times in the table, and
this may be necessary depending on how the instances and geometries in the
scene are setup. Finally, it is also possible to use the instance and geometry
IDs available in the shaders to perform indirect access into other tables
containing the scene data.


Ray Generation
~~~~~~~~~~~~~~~~~~

The Ray Generation shader record consists of a single function referring to the
ray generation shader to be called, along with any desired embedded parameters
for the function. While some parameters can be passed in the shader record, **for
parameters that get updated each frame (e.g., the camera position) it is better
to pass them separately through a different globally accessible buffer**. While
multiple ray generation shaders can be written into the table, only one can be
called for a launch.

Hit Group
~~~~~~~~~~~~

Each Hit Group shader record consists of a Closest Hit shader, Any Hit shader
(optional) and Intersection shader (optional), followed by the set of embedded
parameters to be made available to the three shaders. As the hit group which
should be called is dependent on the instance and geometry which were hit and
the ray type, the indexing rules for hit groups are the most complicated. The
rules for hit group indexing are discussed in detail below.

Hit Group Shader Record Index Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main point of difficulty in setting up the SBT and scene geometry is
understanding how the two are coupled together, i.e., if a geometry is hit by a
ray, which shader record is called? The shader record to call is determined by
parameters set on the instance, trace ray call, and the order of geometries in
the bottom-level acceleration structure. These parameters are set on both the
host and device during different parts of the scene and pipeline setup and
execution, making it difficult to see how they fit together.

::

   HG=&HG[0]+(HGstride×(Roffset+Rstride×GID+Ioffset))(1)

[Yes, that is a good way to think off it as can then write a CPU side
 dumper for the SBT table]


OptiXInstance is 5 quads (5*16 = 80 bytes = 5 quads) 
------------------------------------------------------

::

    echo $(( 12*4 + 6*4 + 1*8  ))

    // for alignment count quads, the instance corresponds to 5 quads (quad is size of float4, ie 16bytes) 
    // OptixTraversableHandle is 64bit so it takes two slots in the last quad and the pad[2] 
    // makes up the 16bytes of the last quad

::

     069 /// Traversable handle
     070 typedef unsigned long long OptixTraversableHandle;

     530 typedef struct OptixInstance
     531 {
     532     /// affine world-to-object transformation as 3x4 matrix in row-major layout
     533     float transform[12];
     534 
     535     /// Application supplied ID. The maximal ID can be queried using OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID.
     536     unsigned int instanceId;
     537 
     538     /// SBT record offset.  Will only be used for instances of geometry acceleration structure (GAS) objects.
     539     /// Needs to be set to 0 for instances of instance acceleration structure (IAS) objects. The maximal SBT offset
     540     /// can be queried using OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_SBT_OFFSET.
     541     unsigned int sbtOffset;
     542 
     543     /// Visibility mask. If rayMask & instanceMask == 0 the instance is culled. The number of available bits can be
     544     /// queried using OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK.
     545     unsigned int visibilityMask;
     546 
     547     /// Any combination of OptixInstanceFlags is allowed.
     548     unsigned int flags;
     549 
     550     /// Set with an OptixTraversableHandle.
     551     OptixTraversableHandle traversableHandle;
     552 
     553     /// round up to 80-byte, to ensure 16-byte alignment
     554     unsigned int pad[2];
     555 } OptixInstance;
     556 




Presentation covering OptiX 7 and SBT Description in Great Detail
------------------------------------------------------------------

* https://developer.nvidia.com/gtc/2020/video/s21888

  * 67 min

* https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21888-rtx-accelerated-raytracing-with-optix-7.pdf

  * 112 page presentation, p73-99 on SBT  

* ~/opticks_refs/sbt-s21888-rtx-accelerated-raytracing-with-optix-7.pdf
* optix7-sbt 


Notes
~~~~~~~

-35:00
     SBT links together programs, program data and geometry
 
-34:00
     pipeline is for the programs, not the program data   

-32:00
     one level instancing is "free" : done in hardware
     if you have a mix of instances and GAS, just add a dummy instance ove the GAS

-31:50
     attributes use registers and are expensive, do not duplicate the built-ins

-31:14
     typically cheaper to recompute the surface normal than to pass attribute from IN to CH
     (huh: do not follow that, because direction of normal depends on geometry ?)

-30:30
     PRD payload, 2x32bit encoding a 64bit pointer to local struct in raygen program

     * using this can write into the local variables of the raygen program from the CH 
     * the examples in the accompanying code do this 2 payload slots encoding a 64bit pointer

-27:43
    Programs comopiled at ProgramGroup granularity

-26:21
    Pipeline hooks up the graph of programGropys, not the data : that comes later

-24:40
    That Tricky Thing : SBT 

-23:07
    SBT specifies which program gets called - with which data - when a given ray hits a given object.

-22:41
    programmatic mapping using a formula based on:

    a) which of the build inputs for a GAS was hit
    b) numSBTRecords in GAS and sbtOffsets in IAS
    c) index and stride from trace call  

    outcome is an integer index into the SBT, specifying 
    the hitprogram group and program data to be executed 

-21:53
    problem: you have to get **four** different things to match:

    a) order of build inputs in GAS(es)
    b) numSBTRecords and sbtOffsets in ASes build inputs
    c) index and stride values passed to optixTrace
    d) the size and order of elements in your SBT 

    Of those four, "d" is the most flexible (and fully under your control) 
    -> this is where you have to make it all match 

    * get this wrong means rays will call the wring programs with wring data
 
-21:18
    SBT : linear array of "SBT Records"

    Each SBT Record
   
    * has a "header" which specifies which program group to execute 
      (stores function pointers, like a virtual method table)
    * has a "payload" or "body" that gets passed to programs as "program data"

      * you choose what to put there
      * has to have the same size (pad smaller elements) 
      * has to satisfy some alignment criteria
      * pointers must point to device memory 
      * programs need to be able to digest the SBT record body  
 

-20:17
   There are actually three such arrays of SBT records for:

   * pointers to RayGen programs + RayGen data
   * pointers to Miss programs + Miss data
   * pointers to HitGroup programs + HitGroup data    (HitGroup: IN/CH) 

   The first two are simple. Usually get problems with the HitGroup SBT.
  
-19:07
   Building/using the SBT involves six steps:

   a) determine the size and order of SBT elements
   b) "build" the SBT entries
   c) upload SBT arrays to device
   d) specify SBT to use during launch
   e) specify "stride" and "offset" to use for each optixTrace call
   f) access SBT data in shader programs

-18:54
   Step b) build

   1. plant "PG function pointers" into header of SBT entry : optixSbtRecordPackHeader(programGroupToUse, sbtRecord.header)
   2. copy the data into body of SBT entry

-18:20
   Step c) upload

   Setup CPU side OptixShaderBindingTable sbt, fill in
   pointers, stride and count for each of the three SBT (raygen, miss, hitgroup) 

   * stride must be a multiple of required alignment 
   * miss, raygen, hitgroup can each have different size 
     but all elements of each array have the same size 


-17:26
   Step d) launch with particular sbt

   * sbt struct is in host mem but arrays it points to are in device mem


-17:13
   Step e) specify stride and offset during launch 

   In raygen, assume ONE ray tyoe

   int sbtStride = 1 ; // num ray types
   int sbtOffset = 0 ; // the ray type
   int sbtMissIndex = 0 ; // the ray type -> which miss program to call when a miss happens

   optixTrace(traversable, ray.origin, ..., sbtOffset,sbtStride,sbtMissIndex, ... )

-16:10
   Assume we traced a slide (with given stride/offset values)...

   * and it hit a triangle/prim in a given instance
   * of a given GAS
   * and that this triangle belongs to the i-th build input for this GAS

   ::

       ias
          i0
             gas0
                bi0 (numSBT=1)
                bi1 (numSBT=1)
          i1
             gas1
                bi0 (numSBT=1)
                bi1 (numSBT=1) 
                bi2 (numSBT=1)
          i2
             gas0
                bi0 
                bi1 
          i3
             gas1
                bi0 
                *bi1*    i3.sbtOffset
                bi2
          i4
             gas0
                bi0 
                bi1 
          i5
             gas1
                bi0 
                bi1 
                bi2
 


          gas0


-15:51
    SBT : what happens under the hood

    a) get the sbtOffset value from the instance we hit -> instOffset 
    b) compute a second offset based on which build input in the GAS we hit -> gasOffset
       eg: 

       * if the GAS has build inputs : A,B,C,D each with numSBT=1 
       * and the prim we hit came from build input C
       * then the GASoffset will be num_A+num_B = 1 + 1 = 2  (for A it would be 0) 

       * Q: what about multiple GAS ?

-14:44

    c) Take sbtOffset and sbtStride from optixTrace call
    d) compute final SBT index::

       int sbtIdx = ray.sbtOffset + instOffset + ray.sbtStride*blasOffset 
       void* addr = (sbtIdx)*sbt.sbtIndexStrideInBytes + sbtBase

    e) call program packed in sbt[sbtIdx].header with data sbt[sbtIdx].body 


-14:22
    So, what about the order of SBT(HitGroup) elements

    Basically, order has to match what this formula computes:

    * all entries from same BLAS/GAS should be in one block 
    * all entries from same BLAS/GAS must match order of build inputs
    * instance sbtOffsets point to where in the SBT the referenced BLAS/GAS starts


-13:18
    eg one BLAS/GAS G with meshes A,B,C one instance I of G 

    * SBT contains { sbtRecord(A), sbtRecord{B}, sbtRecord{C} }
    * I.sbtOffset = 0 
    * ray.sbtOffset = 0, ray.sbtStride = 1 

-12:14
    Another example
  
    * one BLAS G with {A,B,C}
    * 2nd BLAS H woth {D,E}
    * one instance IG (of G) and one IH(of H) in same IAS

    Then:

    * SBT is {A,B,C,D,E}  (first all inputs from G, then all from H)
    * IG.sbtIffset : 0  (offset if A, first entry in G, in the SBT)
    * IH.sbtOffset : 3  (offset of D, first entry in H, in the SBT)

    Note : could have swapped order to {D,E,A,B,A}
    so long as contiguous within each BLAS/GAS

-11:51
    More complicated

    * G = {A,B,C}    H = {E,F}
    * Two instances IG0, IG1 in one IAS + ine instance IH in another IAS

    * SBT is still {A,B,C,D,E}  (build inputs for BLAS/GASes didnt change)
    
    * IG0.sbtOffset : 0
    * IG1.sbtOffset : 0   (both same as same BLAS/GAS with same SBT records)  

    * IH.sbtOffset : 3 (still 3, no matter what IASes there are)  

    [so in this usage pattern the SBT is acting more as an index to the build inputs]
    [that is advantageous for a small SBT, but means instance specifics needs to be handled 
    some other way]


-10:40
     For different ray types to execute different programs
     for same mesh -> need multiple different SBT entries for each mesh (one per ray type)

     Different ways of doing this, "best known method"

     * create N successive SBT entries for each build input (N=num ray types) 
     * adjust the instance offsets
     * pass offset=rayType, stride=N to optixTrace 

-10:02
     Example G={A,B,C} H={E,F} and IG,IH again, but with 2 ray types
    
     * SBT : {A0,A1,B0,B1,C0,C1,E0,E1,F0,F1} 
     * bodies of those entries typically the same
     * optixTrace offset=rayType, stride=2

-08:29
     CAREFUL pitfall: rayStride does not get multiplied into inst.sbtOffset
     
     * so IH.sbtOffset must now change to 6 (from 3)  
     * BLAS/GAS build input for A:F still all numSBTRecords=1 
       because that does get multipled
    
       * think of numSBTRecords as "1 *set* of entries per mesh"

-05:18
     optixLaunch is async, need to cudaStreamSynchronize(stream) to control 
  
     * frame may not be ready until after Synchronize(stream)
       
-04:49
     launch param 

     * fast, lightweight way of passing data 
     * per-frame changing values without SBT rebuild
     * can have multiple launches in flight in parallel 

       * [hmm so each launch gets its own constant memory block] 





    





Dynamic Advice from Vishal
-------------------------------

Regarding the dynamic size of data inferred at runtime.
SBT data lives in GPU global memory like cudaMalloc() 
So, to allow dynamic size is by doing:
 
1. cudaMalloc(&d_data, size);  // size is dynamic
2. packing the pointer in optixLaunch() params. in this you can set params.<var> = d_data to the allocated data.

OR

1. cudaMalloc(&d_data, size);  // size is dynamic
2. then set the pointer d_data to a variable in SBTRecord data structure. So::

    struct HitGroup_CSG_Data
    {
        float *values = d_data;
    };


optix_stubs.h
--------------

::

    // The function table needs to be defined in exactly one translation unit. This can be
    // achieved by including optix_function_table_definition.h in that translation unit.
    extern OptixFunctionTable g_optixFunctionTable;


optix_7_types.h
----------------

::

      69 /// Traversable handle
      70 typedef unsigned long long OptixTraversableHandle;


dependencies between the headers
-----------------------------------

::

    optix.h
        optix_device.h (ifdef __CUDACC__)
            optix_7_device.h

        optix_host.h
            optix_7_host.h

    optix_function_table_definition.h   : plants g_optixFunctionTable
        optix_function_table.h
            optix_types.h
                optix_7_types.h

    optix_stubs.h
        optix_function_table.h          : also plants g_optixFunctionTable
            optix_types.h
                optix_7_types.h

    optix_stack_size.h
        optix.h 


Refs
------

* no multi-GPU, no mem mamagement : all that burden is shifted to application
  (Vulkanization of OptiX 6) 
* but its thread safe 

* Introduced Aug 2019 : OptiX 7 API is not backwards compatible


* https://news.developer.nvidia.com/the-nvidia-optix-sdk-release-7-2/
* https://forums.developer.nvidia.com/t/optix-7-2-release/156619
* https://forums.developer.nvidia.com/t/optix-7-1-release/139962
* https://forums.developer.nvidia.com/t/optix-7-breaking-changes/156801/4


* https://github.com/NVIDIA/OptiX_Apps

* https://github.com/owl-project/owl
* https://www.willusher.io/graphics/2019/11/20/the-sbt-three-ways
* https://github.com/Twinklebear/ChameleonRT






https://developer.nvidia.com/designworks/optix/download







* https://www.nvidia.com/en-us/gtc/session-catalog/?search=OptiX
* https://developer.nvidia.com/gtc/2020/video/s21904
* https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21904-new-features-in-optix-7.pdf

  GTC 2020 (March): New Features in OptiX 7


* https://developer.nvidia.com/gtc/2020/video/s21425 

  ESI Group Report, on OptiX 7 from 41 min
  GTC 2020: Leveraging OptiX 7 for High-Performance Multi-GPU Ray Tracing on Head-Mounted Displays
   




 





* https://news.developer.nvidia.com/optix-7-delivers-new-levels-of-flexibility-to-application-developers/
* https://devtalk.nvidia.com/default/topic/1058310/optix/optix-7-0-release/
* https://devtalk.nvidia.com/default/topic/1061831/optix/optix-debugging-and-profiling-tools/
* https://devtalk.nvidia.com/default/topic/1058577/optix/porting-to-optix-7/

  Detlef : The OptiX 7 API is completely different and the host code effectively requires a rewrite


* https://raytracing-docs.nvidia.com/optix7/index.html
* https://raytracing-docs.nvidia.com/optix7/guide/index.html#introduction#


* https://gitlab.com/ingowald/optix7course
* ~/opticks_refs/SIG19_OptiX7_Main_Talk.pdf


* http://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.5.pdf

* ~/opticks_refs/unofficial_RayTracingGems_v1.5.pdf


Projects Using OptiX to look into
------------------------------------

* https://github.com/BlueBrain/Brayns


OptiX 7.3 (April 2021) 
----------------------------------------------------------

* Requires NVIDIA R465.84 driver or newer. You may need a Beta Driver for certain operating systems.
* ~/opticks_refs/OptiX_Release_Notes_7.3.pdf



OptiX 7.1.0 Release Notes (June 2020)
-----------------------------------------

OptiX 7.1.0 requires that you install a r450+ driver.

* Windows 7/8.1/10 64-bit
* Linux RHEL 4.8+ or Ubuntu 10.10+ 64-bit



Profiling
------------

GTC 2020: What the Profiler is Telling You: How to Get the Most Performance out of Your Hardware
* https://developer.nvidia.com/gtc/2020/video/s22141


NVIDIA Collective Communications Library (NCCL)
--------------------------------------------------

* https://developer.nvidia.com/nccl
* TODO: try this out using CuPy 

DASK
-----

* https://dask.org/



Best Practices: Using NVIDIA RTX Ray Tracing (August 2020)
------------------------------------------------------------

* https://developer.nvidia.com/blog/best-practices-using-nvidia-rtx-ray-tracing/

*  It’s more efficient to handle sky shading in the miss shader rather than in
   the hit shader for the geometry representing the sky.


OpenMC Particle Transport OptiX6 Port
----------------------------------------

* https://sc19.supercomputing.org/proceedings/workshops/workshop_files/ws_pmbsf102s2-file1.pdf
* ~/opticks_refs/Bristol_OpenMC_Particle_Transport_OptiX6_port_ws_pmbsf102s2-file1.pdf



SIGGRAPH 2019 Videos
------------------------

* https://developer.nvidia.com/events/recordings/siggraph-2019

* https://www.nvidia.com/en-us/events/siggraph/schedule/

  * links to more talk videos

* http://www.realtimerendering.com/raytracing/roundup.html


Resources
-----------

* https://gitlab.com/ingowald/optix7course


Gems
--------

* The Iray Light Transport Simulation and Rendering System, A Keller et.al (on arxiv)


RTX Beyond Ray Tracing 
-----------------------

* https://www.willusher.io/publications/rtx-points


SIGGRAPH 2019
--------------

* https://sites.google.com/view/rtx-acc-ray-tracing-with-optix

* https://docs.google.com/document/d/1GKMpK6AjIQsNMPgzdpDtBzEWcJSSZZgB5iLJl7Zlp50/edit

* https://drive.google.com/file/d/1wSz6wTS05YGk6tQOM1l1ubzznmhPo1lH/view

  (99 pages) Ingo Wald, Tutorial OptiX 7, Step by step



Steve Parker SIGGRAPH 2019 Video
----------------------------------

* https://devtalk.nvidia.com/default/topic/1062216/optix/optix-talks-from-siggraph-2019/

16:05
     Sounds like OptiX7 not "fork", its the future

     * explicit memory management (standard CUDA)
     * explicit build AS (accel structures)
     * explicit multi-GPU (up to application)
     * no state on host corresponding to scene

       * zero bytes on host
       * improves startup
       * reduce host resources

19:49
     RAY Tracing APIs : 
 
     * Microsoft DXR (DirectX 12) https://devblogs.microsoft.com/directx/announcing-microsoft-directx-raytracing/
     * NVIDIA VKRay (Vulkan) https://developer.nvidia.com/rtx/raytracing/vkray
     * fat OptiX 1-6
     * thin OptiX 7 


     * sustainable APIs 


20:34
     Buffers -> CUDA Pointers 

     Variables -> shader binding table

     Global scoped variables -> Launch parameters

         * enables overlapping of launches asynchronously 

     Semantic variables -> query functions to access internal state (current ray etc..)

     Amorphous programs -> Tagged program types 

     Geometry Group -> Geometry AS (primitives: triangles or programmable) 

     Group -> Instance AS

         * culling and masking availble on Instance basis

     Transform -> Just Input to Instance AS build


24:21
     OptiX 6 : still continuing to support 

26:55
    3D DAZ STUDIO : IRAY now built on OptiX7     
27:10
     People from NVIDIA worked to integrate OptiX7 into Blender



David Hart, "OptiX Performance Tools and Tricks" SIGGRAPH 2019
-----------------------------------------------------------------

* https://developer.nvidia.com/siggraph/2019/video/sig915-vid

12:25 BVH flatten scene

   Example : modelling two cars with wheels, 

   * should you instance the wheels and the cars or just copy the wheels into car instances 
   * its faster to keep it flatter, ie copy the wheels into two car instances 

   * Hardware Instancing : your first level is free!

     * ie it can handle one level of descent into a smaller BVH 

   * flatten if you can, ie if you have the memory  
   * better to avoid overlapped BVHs

   BVH : one level of instancing comes for free (ie RT cores handle it)


16:24 Reduce memory access

    Quadro RTX 6000 memory bandwidth : 672 GB/s
    *IF* 10 GRays/s is goal, then :

    672 GB/s /  10 GRays/s = 67.2 bytes/ray  (includes traversal)

    => Care must be taken to stay in this budget
    => way to calculate the upper bound of the ray tracer
  
    ::

       BUT : for me, I dont need to pull all the photons back to 
       the GPU, only interested in those that hit PMTs : so I think
       this memory bandwidth should not be a problem

18:01 8 32-bit registers for payload and attribs



Multi-GPU Scheduling 
-----------------------

* https://github.com/owensgroup/mgpuscheduler




GTC Silicon Valley 2019 ID:S9768 : New Features in OptiX 6.0 
----------------------------------------------------------------

* https://developer.nvidia.com/gtc/2019/video/S9768/video

20:17
    Performance with RT cores

                               1
    Amdahl's Law :   S =  -------------    
                           (1-p) + p/s

    For RT core speedup to be large, need to be traversal bound
    50:50 shade/traverse  (shade on SM, traverse on RT cores) 
    -> maximum factor 2 speedup

36:39
    rtContextSetUsageReportCallback

39:34
    BVH compaction : can save 1.5x-2x on memory for +10% build time
    on by default, can be switched off    



Build SDK
--------------

Observations

* glfw, glad and imgui are incorporated

::

    [blyth@localhost OptiX_700]$ mkdir SDK.build
    [blyth@localhost OptiX_700]$ cd SDK.build
    [blyth@localhost SDK.build]$ cmake ../SDK
    ... 
    -- Could NOT find OpenEXR (missing: OpenEXR_IlmImf_RELEASE OpenEXR_Half_RELEASE OpenEXR_Iex_RELEASE OpenEXR_Imath_RELEASE OpenEXR_IlmThread_RELEASE OpenEXR_INCLUDE_DIR) (found version "")
    CMake Warning at optixDemandTexture/CMakeLists.txt:62 (message):
      OpenEXR not found (see OpenEXR_ROOT).  Will use procedural texture in
      optixDemandTexture.

    -- Found ZLIB: /usr/lib64/libz.so (found version "1.2.7") 
    -- Found ZlibStatic: /usr/lib64/libz.so (found version "1.2.7") 
    -- Could NOT find OpenEXR (missing: OpenEXR_IlmImf_RELEASE OpenEXR_Half_RELEASE OpenEXR_Iex_RELEASE OpenEXR_Imath_RELEASE OpenEXR_IlmThread_RELEASE OpenEXR_INCLUDE_DIR) (found version "")
    CMake Warning at optixDemandTextureAdvanced/CMakeLists.txt:62 (message):
      OpenEXR not found (see OpenEXR_ROOT).  Will use procedural texture in
      optixDemandTextureAdvanced.

    -- Found OpenGL: /usr/lib64/libOpenGL.so   
    -- Could NOT find Vulkan (missing: VULKAN_LIBRARY VULKAN_INCLUDE_DIR) 


    [blyth@localhost SDK.build]$ make

    [ 18%] Building C object support/GLFW/src/CMakeFiles/glfw.dir/egl_context.c.o
    [ 20%] Linking C shared library ../../../lib/libglfw.so
    [ 20%] Built target glfw
    Scanning dependencies of target glad
    [ 21%] Building C object support/CMakeFiles/glad.dir/glad/glad.c.o
    [ 22%] Linking C shared library ../lib/libglad.so
    [ 22%] Built target glad
    Scanning dependencies of target imgui
    [ 23%] Building CXX object support/imgui/CMakeFiles/imgui.dir/imgui.cpp.o
    [ 25%] Building CXX object support/imgui/CMakeFiles/imgui.dir/imgui_demo.cpp.o
    ...
    [ 31%] Linking CXX static library ../../lib/libimgui.a
    [ 31%] Built target imgui
    ...


SDK 7 examples
--------------------

Some examples require to pick the display device only::

   Caught exception: GL interop is only available on display device, please use
   display device for optimal performance.  Alternatively you can disable GL
   interop with --no-gl-interop and run with degraded performance.

::

    CUDA_VISIBLE_DEVICES=1 ./optixMeshViewer    ## water bottle
    CUDA_VISIBLE_DEVICES=1 ./optixWhitted       ## bubble and checker board
    CUDA_VISIBLE_DEVICES=1 ./optixCutouts       ## Cornell box with cutouts in sphere and cube 
    CUDA_VISIBLE_DEVICES=1 ./optixSimpleMotionBlur  ## blue and red blurs
    CUDA_VISIBLE_DEVICES=1 ./optixPathTracer     ## Cornell box


    ./optixHello   ## green frame
    ./optixSphere  ## purple sphere
    ./optixSphere  ## blue/cyan/magenta triangle
    ./optixRaycasting      ## writes ppm of duck and translated duck
    ./optixDemandTexture           ## red black checkered sphere
    ./optixDemandTextureAdvanced   ## red black checkered sphere

    CUDA_VISIBLE_DEVICES=0 ./optixMultiGPU    ## Cornell box
    CUDA_VISIBLE_DEVICES=1 ./optixMultiGPU
    CUDA_VISIBLE_DEVICES=0,1 ./optixMultiGPU
    ## all work, but 0,1 has funny checker pattern on image 


CUcontext : is specific to a thread
-------------------------------------

* https://stackoverflow.com/questions/7534892/cuda-context-creation-and-resource-association-in-runtime-api-applications


optixProgramGroup PG and SBT
-------------------------------

* PG is just the bare program
* SBT collects records for each program that hold associated data 
  and reference to the program


github search for optix7
----------------------------

* https://github.com/search?q=optix7
* https://github.com/Hurleyworks/Optix7Sandbox

  mesh instancing example (gltf duck) 

* https://github.com/SpringPie/mov_ass3/tree/master/lighthouse2

* https://github.com/jbikker/lighthouse2/blob/master/lib/RenderCore_Optix7Filter/rendercore.cpp


CUSTOM_PRIMITIVES for analytic
------------------------------------

::

    [blyth@localhost SDK]$ find . -type f -exec grep -H CUSTOM_PRIMITIVES {} \;
    ./optixWhitted/optixWhitted.cpp:    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    ./optixSphere/optixSphere.cpp:            aabb_input.type                    = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    ./optixDemandTextureAdvanced/optixDemandTexture.cpp:    aabb_input.type                    = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    ./optixSimpleMotionBlur/optixSimpleMotionBlur.cpp:        sphere_input.type                     = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    ./optixCutouts/optixCutouts.cpp:        sphere_input.type                    = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    ./optixDemandTexture/optixDemandTexture.cpp:            aabb_input.type                    = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;



dhart : Writing to Buffer
----------------------------

* https://forums.developer.nvidia.com/t/how-to-write-from-closesthit-to-a-device-buffer/110361

I recommend studying the OptiX 7 example called “optixRaycasting”. This sample
is structured to write the ray tracing results to a buffer, which is then
processed by a separate CUDA kernel.

This should give you some ideas of how to handle your payload and the mechanics
of writing to a buffer. If it doesn’t answer your questions, please write back
and we can offer more guidance.

Nsight Compute, Nsight Systems, and Nsight VSE (on Windows), should all work
reasonably well with OptiX 7, as long as you’re using a very recent driver.
cuda-gdb works on Linux, though it’s not as well supported as Nsight.

You can use atomics to prevent two threads from writing to the same memory
address. The best advice for performance is to try hard to avoid needing
atomics, but you can use them if you need. There is a bit more information
about what is allowed here:

https://raytracing-docs.nvidia.com/optix7/guide/index.html#program_pipeline_creation#program-input


EOU
}


optix7-prefix-default(){ echo $OPTICKS_PREFIX/externals/OptiX_700 ; }
optix6-prefix-default(){ echo $OPTICKS_PREFIX/externals/OptiX_650 ; }

optix7-prefix(){ echo ${OPTICKS_OPTIX7_PREFIX:-$(optix7-prefix-default)} ; }
optix6-prefix(){ echo ${OPTICKS_OPTIX6_PREFIX:-$(optix6-prefix-default)} ; }

#optix7-realprefix(){ 
#  local prefix=$OPTICKS_OPTIX_PREFIX 
#  [ -L "$prefix" ] && prefix=$(readlink $OPTICKS_OPTIX_PREFIX) 
#  echo $prefix
#} 
#  no longer works as the links are relative 
#optix7-prefix(){ echo $(dirname $(optix7-realprefix))/OptiX_700 ; }
#optix6-prefix(){ echo $(dirname $(optix7-realprefix))/OptiX_650 ; }


optix7-icd(){ cd $(optix7-prefix)/include ; }
optix7-cd(){ cd $(optix7-prefix)/SDK ; }
optix7-dcd(){ cd $(optix7-prefix)/doc ; }

optix6-icd(){ cd $(optix6-prefix)/include ; }
optix6-cd(){ cd $(optix6-prefix)/SDK ; }
optix6-dcd(){ cd $(optix6-prefix)/doc ; }


optix7-pdf-(){ echo $(optix7-prefix)/doc/OptiX_Programming_Guide_7.0.0.pdf ; }
optix6-pdf-(){ echo $(optix6-prefix)/doc/OptiX_Programming_Guide_6.5.0.pdf ; }

optix7-pdf(){ open $($FUNCNAME-) ; }
optix6-pdf(){ open $($FUNCNAME-) ; }

optix7-html(){ open https://raytracing-docs.nvidia.com/optix7/index.html ; }
optix7-guide(){ open https://raytracing-docs.nvidia.com/optix7/guide/index.html ; }


# open- is from env-
optix6-p(){ open- ; open-page $(( 8 + ${1:-0} )) $(optix6-pdf-) ; }
optix7-p(){ open- ; open-page $(( 4 + ${1:-0} )) $(optix7-pdf-) ; }

optix7-g(){ optix7-cd ; find . -name '*.cpp' -o -name '*.h' -exec grep -Hi ${1:-texture} {} \+ ; }
optix7-l(){ optix7-cd ; find . -name '*.cpp' -o -name '*.h' -exec grep -li ${1:-texture} {} \+ ; }
optix6-g(){ optix6-cd ; find . -name '*.cpp' -o -name '*.h' -exec grep -Hi ${1:-texture} {} \+ ; }
optix6-l(){ optix6-cd ; find . -name '*.cpp' -o -name '*.h' -exec grep -li ${1:-texture} {} \+ ; }


optix7-info(){ cat << EOI

   OPTICKS_PREFIX       : $OPTICKS_PREFIX
   OPTICKS_OPTIX_PREFIX : $OPTICKS_OPTIX_PREFIX

   optix7-realprefix : $(optix7-realprefix)      # obtained with readlink
   optix7-prefix     : $(optix7-prefix)
   optix7-pdf-       : $(optix7-pdf-)


EOI
}



optix7-apps(){
   local dir=/tmp/$USER/opticks
   mkdir -p $dir && cd $dir    
   [ ! -d OptiX_Apps ] && git clone https://github.com/NVIDIA/OptiX_Apps
}



