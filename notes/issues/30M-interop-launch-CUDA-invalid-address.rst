30M-interop-launch-CUDA-invalid-address
================================================

Context
---------

* :doc:`tboolean-generateoverride-photon-scanning`


Observations
------------------

* running in "--compute --cvd 1 --rtx 1" (TITAN RTX) succeeds with 30M : with incredible 7x RTX performance leap

* DONE: revived production running mode which avoids writing the large debug arrays 
 
  * :doc:`big-running-causing-disk-space-pressure`
  * :doc:`revive-production-running-mode` 

* DONE : push this higher, see if can reach the ceiling of 100M, yep reached it  



ISSUE : Error at launch in default "--interop" with 30M photons
-------------------------------------------------------------------

::

    2019-07-13 14:25:36.739 INFO  [325735] [OpEngine::uploadEvent@117] .
    2019-07-13 14:25:36.740 INFO  [325735] [OpEngine::propagate@126] [
    2019-07-13 14:25:36.740 INFO  [325735] [OpSeeder::seedComputeSeedsFromInteropGensteps@63] OpSeeder::seedComputeSeedsFromInteropGensteps : WITH_SEED_BUFFER 
    2019-07-13 14:25:36.770 INFO  [325735] [OpEngine::propagate@136] ( propagator.launch 
    2019-07-13 14:25:39.253 INFO  [325735] [OPropagator::prelaunch@152] 1 : (0;100000000,1) 
    2019-07-13 14:25:39.253 INFO  [325735] [BTimes::dump@146] OPropagator::prelaunch
                  validate000                  8.2e-05
                   compile000                    5e-06
                 prelaunch000                  2.24632
    2019-07-13 14:25:39.253 INFO  [325735] [OPropagator::launch@171] LAUNCH NOW -
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuMemcpyDtoHAsync( dstHost, srcDevice, byteCount, hStream.get() ) returned (700): Illegal address)
    /home/blyth/opticks/bin/o.sh: line 234: 325735 Aborted                 (core dumped) /home/blyth/local/opticks/lib/OKG4Test --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero --generateoverride -100 --rngmax 100 --nog4propagate --envkey --rendermode +glo




ISSUE : ts box with 100M and 30M : gives CUDA illegal address
---------------------------------------------------------------------

::

    OpticksProfile=ERROR TBOOLEAN_TAG=10  ts box --generateoverride -10  --rngmax 100 --nog4propagate        ##  0.417472

    OpticksProfile=ERROR TBOOLEAN_TAG=10  ts box --generateoverride -10  --rngmax 10  --nog4propagate --compute  

    ## does rngmax make any difference to GPU side ?  YES: it dictates the curandState available 
    ##  compare with compute mode, 


    ta box --tag 10    # had to permit some cmdline and rngmax differences between A and B : due to prior Geant4 -10 with smaller rngmax   


    OpticksProfile=ERROR TBOOLEAN_TAG=20  ts box --generateoverride -20  --rngmax 100 --nog4propagate        ##  0.797826 

    ta box --tag 20    # non-existing tagdir for g4, made ana/ab.py changes to still operate to some extent with missing B 



    OpticksProfile=ERROR TBOOLEAN_TAG=100 ts box --generateoverride -100 --rngmax 100 --nog4propagate 
    OpticksProfile=ERROR TBOOLEAN_TAG=30  ts box --generateoverride -30  --rngmax 100 --nog4propagate

          ## 100M and 30M in default interop mode giving below launch error  


    OpticksProfile=ERROR TBOOLEAN_TAG=30  ts box --generateoverride -30  --rngmax 100 --nog4propagate --compute 



Compare COMPUTE mode
-------------------------
   
* does rngmax make any difference to GPU side ?  YES: it dictates the curandState available 

::

    OpticksProfile=ERROR TBOOLEAN_TAG=10  ts box --generateoverride -10  --rngmax 100 --nog4propagate                             ##  0.417472

    OpticksProfile=ERROR TBOOLEAN_TAG=10  ts box --generateoverride -10  --rngmax 10  --nog4propagate --compute                   ##  0.75913  default of --cvd 0,1 doesnt help 

    OpticksProfile=ERROR TBOOLEAN_TAG=10  ts box --generateoverride -10  --rngmax 10  --nog4propagate --compute --cvd 1 --rtx 1   ##  10M,  0.091752    

    ab.pro
          okp 0.0918         g4p 1060.4709       g4p/okp 11552.3643   


    ta box --tag 10      
          ## adder permit for different compute/interop modes between A and B



    OpticksProfile=ERROR TBOOLEAN_TAG=20  ts box --generateoverride -20  --rngmax 100  --nog4propagate --compute --cvd 1 --rtx 1    ##  20M,  0.175749

    OpticksProfile=ERROR TBOOLEAN_TAG=30  ts box --generateoverride -30  --rngmax 100  --nog4propagate --compute --cvd 1 --rtx 1    ##  30M,  0.251551

    ta box --tag 30 
         ## ValueError: mmap length is greater than file size, PROBABLY CAUSED BY low on disk space on tmp


    OpticksProfile=ERROR TBOOLEAN_TAG=30  ts box --generateoverride -30  --rngmax 100  --nog4propagate --compute --cvd 1 --rtx 1 --production   ##   30M, 0.17618

         ## add production mode which skips the debug array collection and saving   

    ta box --tag 30 
         ## adjust analysis to cope with production output, lacking arrays  


    OpticksProfile=ERROR TBOOLEAN_TAG=30  ts box --generateoverride -30  --rngmax 100  --nog4propagate --compute --cvd 1 --rtx 1 --production --savehit    ## 30M, 0.1836 


    OpticksProfile=ERROR TBOOLEAN_TAG=40  ts box --generateoverride -40  --rngmax 100  --nog4propagate --compute --cvd 1 --rtx 1 --production --savehit    ## 40M, 0.2383 

    ta box --tag 40 

    OpticksProfile=ERROR TBOOLEAN_TAG=50  ts box --generateoverride -50  --rngmax 100  --nog4propagate --compute --cvd 1 --rtx 1 --production --savehit    ## 50M, 0.2852  


    OpticksProfile=ERROR TBOOLEAN_TAG=80  ts box --generateoverride -80  --rngmax 100  --nog4propagate --compute --cvd 1 --rtx 1 --production --savehit    ## 80M   --> OOM

    OpticksProfile=ERROR TBOOLEAN_TAG=80  ts box --generateoverride -80  --rngmax 100  --nog4propagate --compute --cvd 1 --rtx 1 --production --savehit    ## 80M  0.1376

         # after create empty sequence and record buffers, can reach 80M, and times are reduced a lot 


    OpticksProfile=ERROR TBOOLEAN_TAG=100  ts box --generateoverride -100  --rngmax 100  --nog4propagate --compute --cvd 1 --rtx 1 --production --savehit    ## 100M   --> OOM 

    OpticksProfile=ERROR TBOOLEAN_TAG=100  ts box --generateoverride -100  --rngmax 100  --nog4propagate --compute --cvd 1 --rtx 1 --production --savehit    ## 100M   0.27114

        ta box --tag 100 



Contrast with G4 extrapolations
-------------------------------------


Extrapolates:

* 1s for 0.010M
* 100s for 1M       1.67 min 
* 1,000s for 10M      16.66 min        (measured 1060.47s)

*  8,000s for 80M
* 10,000s for 100M    166.66 min   2.77 hrs 
* 100,000s for 1000M  1666.66 min  27.77 hrs 


100M ratios, is this real ?::

    In [1]: 10000/0.27114
    Out[1]: 36881.31592535222

    In [2]: 8000/0.1376
    Out[2]: 58139.53488372093


TODO : detailed profiling of OptiX side : see what the overheads are 
-------------------------------------------------------------------------

* eg OpSeeder which populates the seed buffer providing the connection
  between gensteps and photons

* hmm not relevant for input photons ?


TODO: back to GPU photon generation
--------------------------------------

* Hmm generating >50M input photons on CPU takes quite a while, 
  the point of doing so was for easy aligned OK/G4 debugging : but this
  kind of big running aint very practical with G4.   
    
* So need to do generation on GPU for big running, which is closer to real "production" anyhow.
    
* given experience with empyting "sequence" and "record" buffers this will push performance too, 
  by avoiding the big "source" buffer

* aligned running (and running G4) is a debug activity : not to be done in production ...
  perhaps use OKTest, getting rid for G4 entirely 


TODO : delete the eventdirs option, to avoid mixage 
------------------------------------------------------

* analogous to --deletegeocache  --deletetagdir 

* hmm, the paired G4 and OK event pattern is not very convenient with big running :
  because dont do the G4 

* Rearrange such that everything goes in tagdirs ? Tagdirs originally for 
  multievent. Maybe better to resume that, and use the prefix for 
  campaigns with names like "50M". Using tagdirs for different photon counts
  is not a good approach.

* ini no longer needed?

::

    [blyth@localhost torch]$ l
    total 120
    -rw-rw-r--.   1 blyth blyth 1821 Jul 13 22:36 DeltaVM.ini
    -rw-rw-r--.   1 blyth blyth  144 Jul 13 22:36 OpticksProfileAccLabels.npy
    -rw-rw-r--.   1 blyth blyth   96 Jul 13 22:36 OpticksProfileAcc.npy
    -rw-rw-r--.   1 blyth blyth 4112 Jul 13 22:36 OpticksProfileLabels.npy
    -rw-rw-r--.   1 blyth blyth 1088 Jul 13 22:36 OpticksProfile.npy
    -rw-rw-r--.   1 blyth blyth 2397 Jul 13 22:36 VM.ini
    -rw-rw-r--.   1 blyth blyth 1874 Jul 13 22:36 DeltaTime.ini
    -rw-rw-r--.   1 blyth blyth 2189 Jul 13 22:36 Time.ini
    drwxrwxr-x.   4 blyth blyth 4096 Jul 13 22:36 100
    drwxrwxr-x.   4 blyth blyth 4096 Jul 13 22:36 -100
    drwxrwxr-x.   3 blyth blyth 4096 Jul 13 22:23 80
    drwxrwxr-x.   3 blyth blyth 4096 Jul 13 22:23 -80
    drwxrwxr-x. 108 blyth blyth 8192 Jul 13 22:19 1
    drwxrwxr-x. 108 blyth blyth 8192 Jul 13 22:19 -1
    drwxrwxr-x.   3 blyth blyth 4096 Jul 13 20:59 50
    drwxrwxr-x.   3 blyth blyth 4096 Jul 13 20:59 -50
    drwxrwxr-x.   3 blyth blyth 4096 Jul 13 20:47 40
    drwxrwxr-x.   3 blyth blyth 4096 Jul 13 20:47 -40
    drwxrwxr-x.   4 blyth blyth 4096 Jul 13 20:37 30
    drwxrwxr-x.   4 blyth blyth 4096 Jul 13 20:37 -30
    drwxrwxr-x.   4 blyth blyth 4096 Jul 13 17:02 20
    drwxrwxr-x.   6 blyth blyth 4096 Jul 13 16:55 10
    drwxrwxr-x.   3 blyth blyth 4096 Jul 12 17:50 -10
    drwxrwxr-x.   3 blyth blyth 4096 Jul 10 11:43 200
    drwxrwxr-x.   3 blyth blyth 4096 Jul 10 11:43 -200
    drwxrwxr-x.   2 blyth blyth   29 Jun 24 14:28 0
    [blyth@localhost torch]$ 





TODO : some "hit" analysis 
--------------------------------------






Creating the "debug" buffers "sequence" and "record" as empties, allows reaching 80M and 100M, and close to halves the time  
-------------------------------------------------------------------------------------------------------------------------------------

::

    OpticksProfile=ERROR TBOOLEAN_TAG=80  ts box --generateoverride -80  --rngmax 100  --nog4propagate --compute --cvd 1 --rtx 1 --production --savehit 


Huh difference between these 2.  These event dirs are mixed up a bit with prior running::

    2019-07-13 22:23:04.787 INFO  [179114] [OPropagator::launch@180] 1 : (0;80000000,1) 
    2019-07-13 22:23:04.787 INFO  [179114] [BTimes::dump@146] OPropagator::launch
                    launch001                 0.137608
    ...
    [2019-07-13 22:23:05,858] p182969 {load                :ab.py     :250} INFO     - ] 
    ab.pro
          okp 0.1406         g4p 0.0000          g4p/okp 0.0000        




nvidia-smi during the 80M compute run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* hmm though about trying to run nvidia-smi while running these to see memory list, but the launch is less than 0.5s  
* but seems i got lucky:

::

    blyth@localhost optixrap]$ nvidia-smi
    Sat Jul 13 22:23:03 2019       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 418.56       Driver Version: 418.56       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  TITAN RTX           Off  | 00000000:73:00.0  On |                  N/A |
    | 40%   41C    P2    65W / 280W |  15457MiB / 24189MiB |     24%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  TITAN V             Off  | 00000000:A6:00.0 Off |                  N/A |
    | 33%   48C    P8    28W / 250W |      0MiB / 12036MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                               
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0     13840      G   /usr/bin/X                                   131MiB |
    |    0     15631      G   /usr/bin/gnome-shell                         157MiB |
    |    0    179114      C   /home/blyth/local/opticks/lib/OKG4Test     15157MiB |
    +-----------------------------------------------------------------------------+
    [blyth@localhost optixrap]$ nvidia-smi
    Sat Jul 13 22:23:07 2019       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 418.56       Driver Version: 418.56       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  TITAN RTX           Off  | 00000000:73:00.0  On |                  N/A |
    | 41%   43C    P0    99W / 280W |    290MiB / 24189MiB |     34%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  TITAN V             Off  | 00000000:A6:00.0 Off |                  N/A |
    | 33%   48C    P8    28W / 250W |      0MiB / 12036MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                               
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0     13840      G   /usr/bin/X                                   131MiB |
    |    0     15631      G   /usr/bin/gnome-shell                         157MiB |
    +-----------------------------------------------------------------------------+
    [blyth@localhost optixrap]$ 






80M, 100M compute RTX ON with aligned (input photons) gives OOM with TITAN RTX
---------------------------------------------------------------------------------

* FIXED : by setting debug buffers to zero size in production running mode 

::

    2019-07-13 21:14:15.876 ERROR [58087] [OpticksProfile::stamp@180] _OKPropagator::propagate_0 (129.973,15.1914,23697.2,0)
    2019-07-13 21:14:15.876 INFO  [58087] [OpEngine::uploadEvent@117] .
    2019-07-13 21:14:15.876 ERROR [58087] [OpticksProfile::stamp@180] _OEvent::upload_0 (129.973,0,23697.2,0)
    2019-07-13 21:14:15.877 INFO  [58087] [OContext::createBuffer@767]               source        80000000,4,4 mode : COMPUTE  BufferControl : source : OPTIX_INPUT_ONLY UPLOAD_WITH_CUDA BUFFER_COPY_ON_DIRTY COMPUTE_MODE VERBOSE_MODE 
    2019-07-13 21:14:15.887 INFO  [58087] [OContext::upload@682] UPLOAD_WITH_CUDA markDirty (80000000,4,4)  NumBytes(0) 825032704 NumBytes(1) 64 NumValues(0) 1280000000 NumValues(1) 16{}
    2019-07-13 21:14:15.984 INFO  [58087] [OContext::upload@688] UPLOAD_WITH_CUDA markDirty DONE (80000000,4,4)  NumBytes(0) 825032704 NumBytes(1) 64 NumValues(0) 1280000000 NumValues(1) 16{}
    2019-07-13 21:14:15.984 ERROR [58087] [OpticksProfile::stamp@180] OEvent::upload_0 (130.082,0.109375,28710.8,5013.5)
    2019-07-13 21:14:15.984 INFO  [58087] [OpEngine::propagate@126] [
    2019-07-13 21:14:15.984 ERROR [58087] [OpticksProfile::stamp@180] _OpSeeder::seedPhotonsFromGenstepsViaOptiX_0 (130.082,0,28710.8,0)
    2019-07-13 21:14:15.984 INFO  [58087] [OpSeeder::seedPhotonsFromGenstepsViaOptiX@154] SEEDING TO SEED BUF  
    2019-07-13 21:14:15.988 ERROR [58087] [OpticksProfile::stamp@180] OpSeeder::seedPhotonsFromGenstepsViaOptiX_0 (130.086,0.00390625,29038.4,327.68)
    2019-07-13 21:14:15.988 INFO  [58087] [OEvent::markDirty@203] OEvent::markDirty(source) PROCEED
    2019-07-13 21:14:15.988 INFO  [58087] [OpEngine::propagate@136] ( propagator.launch 
    2019-07-13 21:14:15.988 ERROR [58087] [OpticksProfile::stamp@180] _OPropagator::prelaunch_0 (130.086,0,29038.4,0)
    terminate called after throwing an instance of 'optix::Exception'
      what():  Memory allocation failed (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Out of memory)
    /home/blyth/opticks/bin/o.sh: line 234: 58087 Aborted                 (core dumped) /home/blyth/local/opticks/lib/OKG4Test --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero --generateoverride -80 --rngmax 100 --nog4propagate --compute --cvd 1 --rtx 1 --production --savehit --envkey --rendermode +global,+axis --geocenter --stack 2180 --eye 1,0,0 --test --testconfig autoseqmap=TO:0,SR:1,SA:0_name=tboolean-box_outerfirst=1_analytic=1_csgpath=/tmp/blyth/opticks/tboolean-box_mode=PyCsgInBox_autoobj


::

    2019-07-13 21:05:10.627 INFO  [42970] [OEvent::markDirty@203] OEvent::markDirty(source) PROCEED
    2019-07-13 21:05:10.627 INFO  [42970] [OpEngine::propagate@136] ( propagator.launch 
    2019-07-13 21:05:10.627 ERROR [42970] [OpticksProfile::stamp@180] _OPropagator::prelaunch_0 (167.859,0.00390625,31607.7,0)
    terminate called after throwing an instance of 'optix::Exception'
      what():  Memory allocation failed (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Out of memory)
    /home/blyth/opticks/bin/o.sh: line 234: 42970 Aborted                 (core dumped) /home/blyth/local/opticks/lib/OKG4Test --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero --generateoverride -100 --rngmax 100 --nog4propagate --compute --cvd 1 --rtx 1 --production --savehit --envkey --rendermode +global,+axis --geocenter --stack 2180 --eye 1,0,0 --test --testconfig autoseqmap=TO:0,SR:1,SA:0_name=tboolean-box_outerfirst=1_analytic=1_csgpath=/tmp/blyth/opticks/tboolean-box_mode=PyCsgInBox_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vma



Are non-used record_buffer + sequence_buffer  still being allocated in production running ?
----------------------------------------------------------------------------------------------

::

    090 
     91 // input buffers 
     92 
     93 rtBuffer<float4>               genstep_buffer;
     94 rtBuffer<float4>               source_buffer;
     95 #ifdef WITH_SEED_BUFFER
     96 rtBuffer<unsigned>             seed_buffer ;
     97 #endif
     98 rtBuffer<curandState, 1>       rng_states ;
     99 
    100 // output buffers 
    101 
    102 rtBuffer<float4>               photon_buffer;
    103 #ifdef WITH_RECORD
    104 rtBuffer<short4>               record_buffer;     // 2 short4 take same space as 1 float4 quad
    105 rtBuffer<unsigned long long>   sequence_buffer;   // unsigned long long, 8 bytes, 64 bits 
    106 #endif
    107 
    108 



* added handing in OContext to setup empty debug buffers

::


    OpticksProfile=ERROR OEvent=ERROR OContext=ERROR TBOOLEAN_TAG=1  ts box --generateoverride -1  --rngmax 3 --compute --cvd 1 --rtx 1 --production --savehit    ## 1M   







Without "--cvd" both GPUs are used
---------------------------------------

In "--compute" with no "--cvd" option both GPUs are used by default::

    2019-07-13 16:45:35.992 INFO  [88351] [OContext::InitRTX@250]  --rtx 0 setting  OFF
    2019-07-13 16:45:35.999 INFO  [88351] [OContext::CheckDevices@185] 
    Device 0                        TITAN V ordinal 0 Compute Support: 7 0 Total Memory: 12621381632
    Device 1                      TITAN RTX ordinal 1 Compute Support: 7 5 Total Memory: 25364987904




