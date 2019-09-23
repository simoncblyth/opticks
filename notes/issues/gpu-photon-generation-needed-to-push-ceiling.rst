gpu-photon-generation-needed-to-push-ceiling
================================================

issue
-------

* want to push the limits of how many photons, current max is 100M (now 200M)
* scanning high photon counts is slow because of having to generate
  loadsa input photons on the gpu : this was done for alignment. 

* cudarap rngmax monolithic files not scaling well to large photon counts, 
  needs overhaul to a building block approach to reduce GPU memory and disk 
  space waste


* when not doing G4/alignment can return to GPU photon generation, but how ? 

  * emitter stuff in the geometry ?

::

   emitconfig = "photons:100000,wavelength:380,time:0.0,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55" 



Can I reach the memory limit ceiling ?
-----------------------------------------

::

     4*Tesla GV100   : 128G   
     Quadro RTX 8000 :  48G
     Tesla GV100     :  32G      
     TITAN RTX       :  24G
     TITAN V         :  12G         


* photons and rng_states, only buffers needed in production that scale with num_photons.
* geometry does not scale, gensteps scale but they are tiny, step records not used in production.

::

    photons       : 64 bytes 
    curandState   : 48 bytes 
                   112 bytes   per photon

::

    In [3]: 112*100e6/1e9      ## suggests TITAN V will not reach 100M 
    Out[3]: 11.2

    In [4]: 112*200e6/1e9      ## TITAN RTX may just reach 200M,   got 95% GPU memory warnings while doing this 
    Out[4]: 22.4

    In [5]: 112*400e6/1e9      ## Quadro RTX 8000 may just reach 400M
    Out[5]: 44.8

    In [6]: 112*1000e6/1e9     ## 4*Tesla GV100 should manage 1000M : but whether OptiX 6.5 has the smarts is unclear
    Out[6]: 112.0


* needs development to enable multiple curandState files to be used
  in order to reach the 1000M, this would actually be convenient for 
  smaller usage too : as currently have to duplicate content and disk usage
  across the file sizes 

* even with this its unclear if OptiX 6.5 is clever enough to
  handle overlarge buffers 


* :doc:`curandState-persisting-needs-overhaul-for-better-memory-efficiency`



Gold CUDARap TITAN RTX
------------------------

::

    [blyth@localhost cudarap]$ cudarap-prepare-rng-500M
    2019-09-23 20:12:56.492 INFO  [381582] [main@54]  work 500000000 max_blocks 128 seed 0 offset 0 threads_per_block 256 cachedir /home/blyth/.opticks/rngcache/RNG
    2019-09-23 20:12:56.496 INFO  [381582] [allocate_rng_wrapper@68]  items  500000000 items/M 500 sizeof(curandState) 48 nbytes 24000000000 nbytes/M 24000
    Cuda error in file '/home/blyth/opticks/cudarap/cuRANDWrapper_kernel.cu' in line 78 : out of memory.


* observation of nvidia-smi : looks like cudarap is using TITAN V 



how to proceed
------------------

* get some old tboolean which do not use emitconfig running 
* find the old way of configuring photon sources

::

   tboolean-interlocked --nog4


* interlocked is heavy, created boxx which uses torchconfig 


* see documentation in tboolean : Configuring Photon Sources 
* looks like torchconfig still being passed as argument, 
  but is ignored when an emitter is found in the geometry 


::

    [blyth@localhost ana]$ tboolean-
    [blyth@localhost ana]$ tboolean-torchconfig
    type=disc_photons=100000_mode=fixpol_polarization=1,1,0_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,599_target=0,0,0_time=0.1_radius=300_distance=200_zenithazimuth=0,1,0,1_material=Vacuum_wavelength=500



* TODO: make torchconfig honour generateoverride


::

    torchconfig
    ~~~~~~~~~~~

    opticksnpy/TorchStepNPY 
        parses config string and encodes params into genstep buffer for copying to GPU 

    optixrap/cu/torchstep.h
        GPU side OptiX generation of photons from the genstep buffer, this 
        works by throwing two random numbers (ranges from zeaz:zenithazimuth)
        that are used in different ways based upon genstep params  

    cfg4/CTorchSource
        CPU side G4 generation of photons from the genstep buffer, actually the TorchStepNPY instance


    Deficiencies of torchconfig:

    1. get bizarre results when the torch positions are outside the container that 
       defines the Opticks domain, forcing manually tweaking of the 
       torch positions for different containers : the problem with this
       is that it is then not possible to reproduce a prior torch setup via a 
       torchname for example. 

       *emitconfig* solves this issue by decoupling from position, but 
       is currently limited to all sheet/face generation.




Enabling generateoverride to work with torchconfig
------------------------------------------------------

::

    197 void GenstepNPY::setNumPhotons(unsigned int num_photons)
    198 {
    199     m_ctrl.w = num_photons ;
    200 }



* did this in OpticksGen::makeTorchstep

::

    tboolean-;tboolean-interlocked --nog4 --generateoverride 100 
    tboolean-;tboolean-interlocked --nog4 --generateoverride -10 --rngmax 10
        ## visualizing 10M photons works OK on TITAN RTX



scan-px-0
--------------

::

    [blyth@localhost npy]$ SCAN_VERS=0 scan-px-
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_0_1M --generateoverride 1000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 3 --cvd 1 --rtx 0
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_0_10M --generateoverride 10000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 10 --cvd 1 --rtx 0
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_0_20M --generateoverride 20000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 0
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_0_30M --generateoverride 30000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 0
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_0_40M --generateoverride 40000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 0
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_0_50M --generateoverride 50000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 0
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_0_60M --generateoverride 60000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 0
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_0_70M --generateoverride 70000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 0
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_0_80M --generateoverride 80000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 0
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_0_90M --generateoverride 90000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 0
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_0_100M --generateoverride 100000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 0
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_1_1M --generateoverride 1000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 3 --cvd 1 --rtx 1
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_1_10M --generateoverride 10000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 10 --cvd 1 --rtx 1
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_1_20M --generateoverride 20000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 1
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_1_30M --generateoverride 30000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 1
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_1_40M --generateoverride 40000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 1
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_1_50M --generateoverride 50000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 1
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_1_60M --generateoverride 60000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 1
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_1_70M --generateoverride 70000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 1
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_1_80M --generateoverride 80000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 1
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_1_90M --generateoverride 90000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 1
    ts interlocked --oktest --pfx scan-px-0 --cat cvd_1_rtx_1_100M --generateoverride 100000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 100 --cvd 1 --rtx 1


::

    [blyth@localhost ana]$ scan-smry 0 --pfx scan-px 
    INFO:__main__:lookup BashNotes from scan-;scan-px-notes 
     v 0  pfx scan-px-0 
     0
       Gold:TITAN_RTX checking torchconfig based GPU generation of photons with tboolean-interlocked  


    ProfileSmry FromDict:scan-px-0:cvd_1_rtx_0 /home/blyth/local/opticks/evtbase/scan-px-0 1:TITAN_RTX 
    1:TITAN_RTX, RTX OFF
              CDeviceBriefAll : 0:TITAN_V 1:TITAN_RTX 
              CDeviceBriefVis : 1:TITAN_RTX 
                      RTXMode : 0 
        NVIDIA_DRIVER_VERSION : 435.21 
                     name       note  av.interv  av.launch  av.overhd                                             launch 
           cvd_1_rtx_0_1M   MULTIEVT     0.1168     0.1137     1.0271 array([0.1211, 0.1172, 0.1133, 0.1133, 0.1094, 0.1133, 0.1094, 0.1133, 0.1133, 0.1133], dtype=float32) 
          cvd_1_rtx_0_10M   MULTIEVT     1.1298     1.1215     1.0074 array([1.1875, 1.1211, 1.1289, 1.1055, 1.1016, 1.1055, 1.1133, 1.125 , 1.1172, 1.1094], dtype=float32) 
          cvd_1_rtx_0_20M   MULTIEVT     2.2253     2.2160     1.0042 array([2.2266, 2.2148, 2.2109, 2.2148, 2.2148, 2.2109, 2.207 , 2.2305, 2.2148, 2.2148], dtype=float32) 
          cvd_1_rtx_0_30M   MULTIEVT     3.3837     3.3656     1.0054 array([3.3906, 3.4609, 3.4062, 3.375 , 3.3359, 3.3242, 3.3359, 3.332 , 3.3477, 3.3477], dtype=float32) 
          cvd_1_rtx_0_40M   MULTIEVT     4.3872     4.3668     1.0047 array([4.4414, 4.3516, 4.3477, 4.3672, 4.375 , 4.3594, 4.3555, 4.3516, 4.3633, 4.3555], dtype=float32) 
          cvd_1_rtx_0_50M   MULTIEVT     5.7248     5.7020     1.0040 array([5.7109, 5.6914, 5.6953, 5.6914, 5.6836, 5.7266, 5.7109, 5.707 , 5.707 , 5.6953], dtype=float32) 
          cvd_1_rtx_0_60M   MULTIEVT     6.7782     6.7484     1.0044 array([6.7109, 6.9023, 6.75  , 6.7773, 6.7266, 6.7305, 6.75  , 6.7148, 6.7188, 6.7031], dtype=float32) 
          cvd_1_rtx_0_70M   MULTIEVT     7.8880     7.8430     1.0057 array([7.7617, 7.7109, 7.7109, 7.7852, 8.3906, 8.082 , 7.7227, 7.7969, 7.75  , 7.7188], dtype=float32) 
          cvd_1_rtx_0_80M   MULTIEVT     8.8598     8.8293     1.0035 array([8.8867, 8.9062, 8.8164, 8.8008, 8.8047, 8.8008, 8.7969, 8.8008, 8.8242, 8.8555], dtype=float32) 


    ProfileSmry FromDict:scan-px-0:cvd_1_rtx_1 /home/blyth/local/opticks/evtbase/scan-px-0 None 
    None, RTX ?
              CDeviceBriefAll : None 
              CDeviceBriefVis : None 
                      RTXMode : None 
        NVIDIA_DRIVER_VERSION : None 
                     name       note  av.interv  av.launch  av.overhd                                             launch 
    [blyth@localhost ana]$ 




