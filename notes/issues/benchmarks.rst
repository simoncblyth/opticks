benchmarks
==============

Principals

* need running from cache : DONE

* "--compute" mode is what matters 
* time for raytrace snapshots is an obvious metric, eg start from okop/tests/OpSnapTest.cc 

  * need to automate viewpoint and camera params in a non-fragile way (bookmarks are fragile)
    commandline arguments less so

* possibly running without GUI (runlevel 3) can avoid any OpenGL involvement

* use CUDA_VISIBLE_DEVICES 

  1. unset
  2. 0,1   # expect same as unset
  3. 1,0   # expect same as unset
  4. 0
  5. 1

* implement sensitivity to OPTICKS_RTX=0,1 for switching the attribute 
* currently there is an OpenGL way of detecting the GPU for the context, 
  instead need a compute version of that (see UseOptiX) that OContext 
  perhaps holds onto and reports into metadata



First in interop for dev
----------------------------

No obvious change in interop::

    [blyth@localhost optixrap]$ CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=1 OKTest --envkey --xanalytic --target 10000
    [blyth@localhost optixrap]$ OPTICKS_RTX=0 OKTest --envkey --xanalytic --target 10000


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





