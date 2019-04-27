benchmarks
==============


Things to try to speedup analytic
---------------------------------------

1. simplifiy geometry tree : DONE, NO SIGNIFICANT CHANGE
2. change accel builders : DONE, NO SIGNIFICANT CHANGE
3. review the analytic buffers 

   * especially the prismBuffer, why is it INPUT_OUTPUT 
   * its very small : just try to get rid of it 




Titan V and Titan RTX : Effect of RTX execution mode
----------------------------------------------------------------

Comparing raytrace performance of Titan V and Titan RTX 
with a modified JUNO geometry with Torus removed
from PMTs and guidetube by changing the input GDML. 
(OptiX 6.0.0 crashes when attempting to use my quartic 
root finding for the Torus.)

My benchmark metric is the average of five very high resolution 
5120x2880 ~15M pixels raytrace launch times near the JUNO 
chimney with a large number of PMTs in view.

I use three RTX mode variations:

   R0
       RTX off : ordinary software BVH traversal and intersection
   R1
       RTX on : only BVH traversal using RT Cores, intersection in software
   R2
       RTX on + intersection handled with RT Cores using GeometryTriangles (new in OptiX 6) 


Times for triangulated geometry in seconds:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

       .        20190424_203832     metric      rfast      rslow 

                   R2_TITAN_RTX      0.037      1.000      0.250 
                   R1_TITAN_RTX      0.074      2.018      0.505 
       R0_TITAN_V_AND_TITAN_RTX      0.078      2.129      0.533 
                     R2_TITAN_V      0.100      2.722      0.682 
                   R0_TITAN_RTX      0.103      2.810      0.704 
                     R1_TITAN_V      0.116      3.149      0.789 
                     R0_TITAN_V      0.147      3.993      1.000 

Example commandline::

   OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 \
              --embedded --rtx 2 --runfolder geocache-bench --runstamp 1556109512 --runlabel R2_TITAN_RTX


Observations:

* fractions of a second for 15M pixels bodes well 
* TITAN RTX gains a factor of ~3 from R0 to R2 
* TITAN V doesnt have RT cores, but RTX mode still improves its times


Times for analytic geometry in seconsds 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

       .        20190424_204442     metric      rfast      rslow 

       R0_TITAN_V_AND_TITAN_RTX      0.122      1.000      0.188   
                   R0_TITAN_RTX      0.188      1.537      0.289 
                     R0_TITAN_V      0.219      1.790      0.337    
                   R1_TITAN_RTX      0.540      4.420      0.831     
                     R1_TITAN_V      0.650      5.319      1.000 

Example commandline::

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 \
                --embedded --rtx 0 --runfolder geocache-bench --runstamp 1556109882 --runlabel R0_TITAN_V_AND_TITAN_RTX --xanalytic

Observations:

* cost for the exact geometry is about a factor 4 over the approximate triangulated ones
  (I'm happy that my CSG processing does not cost more that that)

* analytic really benefits from the core counts (TITAN V + TITAN RTX) 5120+4680 CUDA cores
  getting into the ballpark of triangulated geometries
  
  * i look forward to trying this benchmark on the GPU cluster nodes  
  
* RTX mode makes analytic times worse : by a factor of 2-3 

  * without using triangles, the only way the RT cores can help
    is with the BVH traversal being done in hardware : the fact 
    that timings get worse by as much as a factor of 3 suggests I should
    try some alternative OptiX acceleration/geometry setups  






With my triangles, ie no --xanalytic
-----------------------------------------

* This is with the torus-less GDML j1808 v3. 
* Note the 14.7M pixels. 
* The metric is launchAVG of five launch times.  
* OFF/ON refers to RTX execution approach
* OPTICKS_KEY OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
* commandline for the first of each group of runs is given as it was the same, the 
  differnence coming from envvars CUDA_VISIBLE_DEVICES and OPTICKS_RTX


::

    [blyth@localhost opticks]$ bench.py $LOCAL_BASE/opticks/results/geocache-bench
     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --runfolder geocache-bench --runstamp 1555926978 --runlabel ON_TITAN_RTX
                    20190422_175618     metric      rfast      rslow 
                       ON_TITAN_RTX      0.056      1.000      0.391 
          OFF_TITAN_V_AND_TITAN_RTX      0.080      1.431      0.560 
                      OFF_TITAN_RTX      0.108      1.923      0.752 
                         ON_TITAN_V      0.117      2.083      0.815 
                        OFF_TITAN_V      0.143      2.557      1.000 

     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --runfolder geocache-bench --runstamp 1555940309 --runlabel ON_TITAN_RTX
                    20190422_213829     metric      rfast      rslow 
                       ON_TITAN_RTX      0.073      1.000      0.503 
          OFF_TITAN_V_AND_TITAN_RTX      0.081      1.109      0.557 
                         ON_TITAN_V      0.116      1.589      0.799 
                      OFF_TITAN_RTX      0.117      1.607      0.808 
                        OFF_TITAN_V      0.145      1.990      1.000 



* RTX speedup should be more by using  optix::GeometryTriangles




/usr/local/OptiX_600/SDK-src/optixGeometryTriangles
--------------------------------------------------------




Finding target volume to snap
-------------------------------

Found a good viewpoint, looking up at chimney::

    CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=1 OKTest --envkey --xanalytic --target 352851 --eye -1,-1,-1        ## analytic
    CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=1 OKTest --envkey --target 352851 --eye -1,-1,-1                    ## tri 

    CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=-1 OpSnapTest --envkey --xanalytic --target 352851 --eye -1,-1,-1 


* target is 0-based 
* numbers listed in PVNames.txt from *vi* in the below are 1-based 
* 352851 is pLowerChimneyLS0x5b317e0 

GNodeLib/PVNames.txt::

    .1 lWorld0x4bc2710_PV
     2 pTopRock0x4bcd120
     3 pExpHall0x4bcd520
     4 lUpperChimney_phys0x5b308a0
     5 pUpperChimneyLS0x5b2f160
    ...

    352847 PMT_3inch_inner1_phys0x510beb0
    352848 PMT_3inch_inner2_phys0x510bf60
    352849 PMT_3inch_cntr_phys0x510c010
    352850 lLowerChimney_phys0x5b32c20
    352851 pLowerChimneyAcrylic0x5b31720
    352852 pLowerChimneyLS0x5b317e0
    352853 pLowerChimneySteel0x5b318b0
    352854 lSurftube_phys0x5b3c810
    352855 pvacSurftube0x5b3c120
    352856 lMaskVirtual_phys0x5cc1ac0



OpSnapTest
-------------

* :doc:`OpSnapTest_review`



Unless I am missing something. 

* perhaps compiling with CC 75 rather than current 70 ?
* also need to check with snap paths across more demanding geometry 

Take a look at a more demanding render over in env- rtow-



Perhaps JIT compilation killing perfermanance for TITAN RTX ?

cmake/Modules/OpticksCUDAFlags.cmake needs to handle a comma delimited COMPUTE_CAPABILITY ?::

     09 if(NOT (COMPUTE_CAPABILITY LESS 30))
     10 
     11    #list(APPEND CUDA_NVCC_FLAGS "-arch=sm_${COMPUTE_CAPABILITY}")
     12    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
     13    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}")
     14 
     15    #list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
     16    # https://github.com/facebookresearch/Detectron/issues/185
     17 
     18    list(APPEND CUDA_NVCC_FLAGS "-O2")
     19    #list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE")
     20    list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
     21 
     22    #list(APPEND CUDA_NVCC_FLAGS "-m64")
     23    #list(APPEND CUDA_NVCC_FLAGS "--disable-warnings")
     24 
     25    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
     26    set(CUDA_VERBOSE_BUILD OFF)
     27 
     28 endif()




After Fixing Several Bugs 
-----------------------------------------------------------------

Bugs included:

* prelaunch doing launch
* mis-configured snap positions

And:

* increasing size 
* finding a region with lots of PMTs
* switch to trianglulated ( no --xanalytic )


::

    [blyth@localhost optixrap]$ t geocache-bench
    geocache-bench is a function
    geocache-bench () 
    { 
        echo "TITAN RTX";
        CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=0 $FUNCNAME-;
        CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=1 $FUNCNAME-;
        echo "TITAN V";
        CUDA_VISIBLE_DEVICES=0 OPTICKS_RTX=0 $FUNCNAME-;
        CUDA_VISIBLE_DEVICES=0 OPTICKS_RTX=1 $FUNCNAME-
    }


::

    geocache-bench- is a function
    geocache-bench- () 
    { 
        type $FUNCNAME;
        local dbg;
        [ -n "$DBG" ] && dbg="gdb --args" || dbg="";
        $dbg OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig "steps=5,eyestartz=-1,eyestopz=-0.5" --size 5120,2880,1 --embedded $*
    }
    2019-04-21 22:53:02.945 INFO  [155128] [BOpticksKey::SetKey@45] from OPTICKS_KEY envvar OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
    2019-04-21 22:53:11.224 INFO  [155128] [OTracer::report@157] OpTracer::snap
     trace_count              5 trace_prep        0.075119 avg  0.0150238
     trace_time         2.24857 avg   0.449713

    2019-04-21 22:53:11.224 INFO  [155128] [BTimes::dump@138] OTracer::report
                  validate000                 0.050209
                   compile000                    7e-06
                 prelaunch000                  1.59024
                    launch000                 0.132858
                    launch001                  0.10317
                    launch002                 0.102913
                    launch003                 0.105186
                    launch004                 0.101064
                    launchAVG                 0.109038
    2019-04-21 22:53:11.224 INFO  [155128] [BMeta::dump@53] Opticks OpTracer::snap
    CUDA_VISIBLE_DEVICES : 1
             OPTICKS_RTX : 0
             OPTICKS_KEY : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
                 CMDLINE :  OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded
    2019-04-21 22:53:11.225 INFO  [155128] [OpTracer::snap@132] )
    geocache-bench- is a function

    2019-04-21 22:53:19.575 INFO  [155416] [BTimes::dump@138] OTracer::report
                  validate000                   0.0517
                   compile000                    8e-06
                 prelaunch000                  1.52944
                    launch000                 0.057163
                    launch001                 0.056131
                    launch002                 0.055519
                    launch003                 0.056188
                    launch004                 0.056055
                    launchAVG                0.0562112
    2019-04-21 22:53:19.576 INFO  [155416] [BMeta::dump@53] Opticks OpTracer::snap
    CUDA_VISIBLE_DEVICES : 1
             OPTICKS_RTX : 1
             OPTICKS_KEY : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
                 CMDLINE :  OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded
    2019-04-21 22:53:19.576 INFO  [155416] [OpTracer::snap@132] )


    2019-04-21 22:53:28.396 INFO  [155678] [BTimes::dump@138] OTracer::report
                  validate000                 0.052362
                   compile000                    9e-06
                 prelaunch000                  1.74231
                    launch000                 0.139875
                    launch001                 0.146404
                    launch002                 0.143448
                    launch003                 0.143731
                    launch004                 0.141017
                    launchAVG                 0.142895
    2019-04-21 22:53:28.396 INFO  [155678] [BMeta::dump@53] Opticks OpTracer::snap
    CUDA_VISIBLE_DEVICES : 0
             OPTICKS_RTX : 0
             OPTICKS_KEY : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
                 CMDLINE :  OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded
    2019-04-21 22:53:37.127 INFO  [155967] [BTimes::dump@138] OTracer::report
                  validate000                 0.051268
                   compile000                    8e-06
                 prelaunch000                  1.47854
                    launch000                 0.113385
                    launch001                 0.117253
                    launch002                 0.116381
                    launch003                 0.116277
                    launch004                 0.118571
                    launchAVG                 0.116373
    2019-04-21 22:53:37.128 INFO  [155967] [BMeta::dump@53] Opticks OpTracer::snap
    CUDA_VISIBLE_DEVICES : 0
             OPTICKS_RTX : 1
             OPTICKS_KEY : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
                 CMDLINE :  OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded
    2019-04-21 22:53:37.128 INFO  [155967] [OpTracer::snap@132] )
    [blyth@localhost sysrap]$ 





