nvidia-quadro-rtx-8000-scan-ph-testing-on-gilda03-silver-precision
=====================================================================


Shakedown
-----------

1. curiously it works with the Geforce driver
2. update opticks and move geocache into ~/.opticks/ 
3. opticks-t succeeds with the usual 2 fails
4. add cudarap-prepare-installcache-100M and run it,  standard cudarap-prepare-installcache only goes to 3M

::

    (base) [blyth@gilda03 ~]$ du -h $(cudarap-rngdir)/*
    4.1G    /home/blyth/local/opticks/installcache/RNG/cuRANDWrapper_100000000_0_0.bin
    440K    /home/blyth/local/opticks/installcache/RNG/cuRANDWrapper_10240_0_0.bin
    126M    /home/blyth/local/opticks/installcache/RNG/cuRANDWrapper_3000000_0_0.bin



profilesmry.py comparison
-------------------------------


* same 70M hit dropping issue 


silver precision with Quadro RTX 8000::

    (base) [blyth@gilda03 ana]$ an ; ip profilesmry.py 
    Python 2.7.16 |Anaconda, Inc.| (default, Mar 14 2019, 21:00:58) 
    Type "copyright", "credits" or "license" for more information.

    IPython 5.8.0 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.
    car       cvd_0_rtx_0_1M npho   1000000 nhit       516 ihit  1937     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_0_1M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_0_20M npho  20000000 nhit     10406 ihit  1921     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_0_20M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_0_30M npho  30000000 nhit     15675 ihit  1913     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_0_30M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_0_40M npho  40000000 nhit     20802 ihit  1922     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_0_40M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_0_50M npho  50000000 nhit     26031 ihit  1920     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_0_50M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_0_60M npho  60000000 nhit     31126 ihit  1927     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_0_60M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_0_70M npho  70000000 nhit      1562 ihit 44814     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_0_70M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_0_80M npho  80000000 nhit      6663 ihit 12006     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_0_80M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_0_90M npho  90000000 nhit     11963 ihit  7523     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_0_90M/torch/OpticksProfile.npy 
    car     cvd_0_rtx_0_100M npho 100000000 nhit     17122 ihit  5840     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_0_100M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_0_10M npho  10000000 nhit      5212 ihit  1918     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_0_10M/torch/OpticksProfile.npy 
    car       cvd_0_rtx_1_1M npho   1000000 nhit       516 ihit  1937     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_1_1M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_1_10M npho  10000000 nhit      5212 ihit  1918     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_1_10M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_1_20M npho  20000000 nhit     10406 ihit  1921     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_1_20M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_1_30M npho  30000000 nhit     15675 ihit  1913     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_1_30M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_1_40M npho  40000000 nhit     20802 ihit  1922     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_1_40M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_1_50M npho  50000000 nhit     26031 ihit  1920     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_1_50M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_1_60M npho  60000000 nhit     31126 ihit  1927     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_1_60M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_1_70M npho  70000000 nhit      1562 ihit 44814     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_1_70M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_1_80M npho  80000000 nhit      6663 ihit 12006     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_1_80M/torch/OpticksProfile.npy 
    car      cvd_0_rtx_1_90M npho  90000000 nhit     11963 ihit  7523     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_1_90M/torch/OpticksProfile.npy 
    car     cvd_0_rtx_1_100M npho 100000000 nhit     17122 ihit  5840     path /home/blyth/local/opticks/tmp/scan-ph-0/evt/cvd_0_rtx_1_100M/torch/OpticksProfile.npy 

    In [1]: 



gold precision with Titan RTX::

    [blyth@localhost ana]$ ip
    Python 2.7.15 |Anaconda, Inc.| (default, May  1 2018, 23:32:55) 
    Type "copyright", "credits" or "license" for more information.

    IPython 5.7.0 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    In [1]: run profilesmry.py
    car       cvd_1_rtx_0_1M npho   1000000 nhit       516 ihit  1937     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_0_1M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_0_10M npho  10000000 nhit      5212 ihit  1918     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_0_10M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_0_20M npho  20000000 nhit     10406 ihit  1921     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_0_20M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_0_30M npho  30000000 nhit     15675 ihit  1913     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_0_30M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_0_40M npho  40000000 nhit     20802 ihit  1922     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_0_40M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_0_50M npho  50000000 nhit     26031 ihit  1920     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_0_50M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_0_60M npho  60000000 nhit     31126 ihit  1927     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_0_60M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_0_70M npho  70000000 nhit      1562 ihit 44814     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_0_70M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_0_80M npho  80000000 nhit      6663 ihit 12006     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_0_80M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_0_90M npho  90000000 nhit     11963 ihit  7523     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_0_90M/torch/OpticksProfile.npy 
    car     cvd_1_rtx_0_100M npho 100000000 nhit     17122 ihit  5840     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_0_100M/torch/OpticksProfile.npy 
    car       cvd_1_rtx_1_1M npho   1000000 nhit       516 ihit  1937     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1_1M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_10M npho  10000000 nhit      5212 ihit  1918     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1_10M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_20M npho  20000000 nhit     10406 ihit  1921     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1_20M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_30M npho  30000000 nhit     15675 ihit  1913     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1_30M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_40M npho  40000000 nhit     20802 ihit  1922     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1_40M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_50M npho  50000000 nhit     26031 ihit  1920     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1_50M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_60M npho  60000000 nhit     31126 ihit  1927     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1_60M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_70M npho  70000000 nhit      1562 ihit 44814     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1_70M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_80M npho  80000000 nhit      6663 ihit 12006     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1_80M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_90M npho  90000000 nhit     11963 ihit  7523     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1_90M/torch/OpticksProfile.npy 
    car     cvd_1_rtx_1_100M npho 100000000 nhit     17122 ihit  5840     path /home/blyth/local/opticks/tmp/scan-ph/evt/cvd_1_rtx_1_100M/torch/OpticksProfile.npy 
    car       cvd_1_rtx_1_1M npho   1000000 nhit       516 ihit  1937     path /home/blyth/local/opticks/tmp/scan-ph-tri/evt/cvd_1_rtx_1_1M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_10M npho  10000000 nhit      5212 ihit  1918     path /home/blyth/local/opticks/tmp/scan-ph-tri/evt/cvd_1_rtx_1_10M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_20M npho  20000000 nhit     10406 ihit  1921     path /home/blyth/local/opticks/tmp/scan-ph-tri/evt/cvd_1_rtx_1_20M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_30M npho  30000000 nhit     15675 ihit  1913     path /home/blyth/local/opticks/tmp/scan-ph-tri/evt/cvd_1_rtx_1_30M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_40M npho  40000000 nhit     20802 ihit  1922     path /home/blyth/local/opticks/tmp/scan-ph-tri/evt/cvd_1_rtx_1_40M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_50M npho  50000000 nhit     26031 ihit  1920     path /home/blyth/local/opticks/tmp/scan-ph-tri/evt/cvd_1_rtx_1_50M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_60M npho  60000000 nhit     31126 ihit  1927     path /home/blyth/local/opticks/tmp/scan-ph-tri/evt/cvd_1_rtx_1_60M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_70M npho  70000000 nhit      1562 ihit 44814     path /home/blyth/local/opticks/tmp/scan-ph-tri/evt/cvd_1_rtx_1_70M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_80M npho  80000000 nhit      6663 ihit 12006     path /home/blyth/local/opticks/tmp/scan-ph-tri/evt/cvd_1_rtx_1_80M/torch/OpticksProfile.npy 
    car      cvd_1_rtx_1_90M npho  90000000 nhit     11963 ihit  7523     path /home/blyth/local/opticks/tmp/scan-ph-tri/evt/cvd_1_rtx_1_90M/torch/OpticksProfile.npy 
    car     cvd_1_rtx_1_100M npho 100000000 nhit     17122 ihit  5840     path /home/blyth/local/opticks/tmp/scan-ph-tri/evt/cvd_1_rtx_1_100M/torch/OpticksProfile.npy 

    In [2]: 


profilesmryplot.py comparison of TITAN RTX and Quadro RTX 8000
---------------------------------------------------------------------

* seems that cannot switch on RTX mode with the Geforce driver, 
  other than no RTX mode advantage the plots (at a high level without close scrutiny) 
  look the same as each other 



after update driver to 430.50 using run file many opticks-t tests shows PTX errors
-------------------------------------------------------------------------------------

After gfw- addition of some sites to make the nvidia downloads page work properly 
can see that the 430.50 driver is extremelyt recent, dated 



* see onvidia- for how the update was done


opticks-t fails

::


    FAILS:  19  / 411   :  Thu Sep 12 16:07:16 2019   
      3  /24  Test #3  : OptiXRapTest.LTOOContextUploadDownloadTest    Child aborted***Exception:     1.64   
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     2.39   
      5  /24  Test #5  : OptiXRapTest.bufferTest                       Child aborted***Exception:     1.63   
      6  /24  Test #6  : OptiXRapTest.textureTest                      Child aborted***Exception:     1.66   
      7  /24  Test #7  : OptiXRapTest.boundaryTest                     Child aborted***Exception:     1.66   
      8  /24  Test #8  : OptiXRapTest.boundaryLookupTest               Child aborted***Exception:     1.68   
      12 /24  Test #12 : OptiXRapTest.rayleighTest                     Child aborted***Exception:     7.09   
      13 /24  Test #13 : OptiXRapTest.writeBufferTest                  Child aborted***Exception:     1.63   
      16 /24  Test #16 : OptiXRapTest.downloadTest                     Child aborted***Exception:     1.70   
      17 /24  Test #17 : OptiXRapTest.eventTest                        Child aborted***Exception:     2.02   
      18 /24  Test #18 : OptiXRapTest.interpolationTest                Child aborted***Exception:     3.29   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.87   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     7.57   
      3  /5   Test #3  : OKOPTest.dirtyBufferTest                      Child aborted***Exception:     1.60   
      4  /5   Test #4  : OKOPTest.compactionTest                       Child aborted***Exception:     1.66   
      5  /5   Test #5  : OKOPTest.OpSnapTest                           Child aborted***Exception:     3.15   
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     10.29  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     26.89  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      10.90  



::

    2019-09-12 16:07:09.845 INFO  [70894] [OpEngine::propagate@155] ( propagator.launch 
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: ptxas application ptx input, line 58; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 61; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 63; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 65; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 67; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 69; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 71; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 73; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 75; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 77; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 1796; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 1798; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 1938; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 1939; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 1941; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 1942; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 1944; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 1945; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 1947; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 1948; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 1950; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 1951; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas fatal   : Ptx assembly aborted due to errors returned (218): Invalid ptx)
    /home/blyth/opticks/bin/o.sh: line 253: 70894 Aborted                 (core dumped) /home/blyth/local/opticks/lib/OKG4Test --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero --generateoverride 10000 --envkey --rendermode +global,+axis --geocenter --stack 2180 --eye 1,0,0 --up 0,0,1 --test --testconfig autoseqmap=TO:0,SR:1,SA:0_name=tboolean-box_outerfirst=1_analytic=1_csgpath=/home/blyth/local/opticks/tmp/tboolean-box_mode=PyCsgInBox_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0_autocontainer=Rock//perfectAbsorbSurface/Vacuum --torch --torchconfig type=disc_photons=100000_mode=fixpol_polarization=1,1,0_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,599_target=0,0,0_time=0.1_radius=300_distance=200_zenithazimuth=0,1,0,1_material=Vacuum_wavelength=500 --torchdbg --tag 1 --anakey tboolean --args --save
    === o-main : /home/blyth/local/opticks/lib/OKG4Test --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero --generateoverride 10000 --envkey --rendermode +global,+axis --geocenter --stack 2180 --eye 1,0,0 --up 0,0,1 --test --testconfig autoseqmap=TO:0,SR:1,SA:0_name=tboolean-box_outerfirst=1_analytic=1_csgpath=/home/blyth/local/opticks/tmp/tboolean-box_mode=PyCsgInBox_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0_autocontainer=Rock//perfectAbsorbSurface/Vacuum --torch --torchconfig type=disc_photons=100000_mode=fixpol_polarization=1,1,0_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,599_target=0,0,0_time=0.1_radius=300_distance=200_zenithazimuth=0,1,0,1_material=Vacuum_wavelength=500 --torchdbg --tag 1 --anakey tboolean --args --save ======= PWD /home/blyth/local/opticks/build/integration/tests RC 134 Thu Sep 12 16:07:16 CST 2019
    echo o-postline : dummy


::

    (base) [blyth@gilda03 PTX]$ head -15 OptiXRap_generated_generate.cu.ptx
    //
    // Generated by NVIDIA NVVM Compiler
    //
    // Compiler Build ID: CL-25769353
    // Cuda compilation tools, release 10.1, V10.1.105
    // Based on LLVM 3.4svn
    //

    .version 6.4
    .target sm_70
    .address_size 64

        // .globl   _Z7nothingv
    .extern .func  (.param .b32 func_retval0) vprintf
    (


Try rebuilding the PTX, and a opticks build::

    oxrap-c
    oxrap-wipe
    om-install

    o
    om--  


All the PTX are rebuilt, but the head looks the same::

    (base) [blyth@gilda03 PTX]$ head -15 OptiXRap_generated_generate.cu.ptx
    //
    // Generated by NVIDIA NVVM Compiler
    //
    // Compiler Build ID: CL-25769353
    // Cuda compilation tools, release 10.1, V10.1.105
    // Based on LLVM 3.4svn
    //

    .version 6.4
    .target sm_70
    .address_size 64

        // .globl   _Z7nothingv
    .extern .func  (.param .b32 func_retval0) vprintf




::

    (base) [blyth@gilda03 ~]$ bufferTest 
    2019-09-12 16:17:32.419 INFO  [80443] [Opticks::init@339] COMPUTE_MODE compute_requested 
    2019-09-12 16:17:32.425 INFO  [80443] [Opticks::initResource@693]  (legacy mode) setting IDPATH envvar for python analysis scripts [/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae]
    2019-09-12 16:17:32.425 FATAL [80443] [Opticks::defineEventSpec@1897]  resource_pfx (null) config_pfx (null) pfx default_pfx cat (null) udet dayabay typ torch tag 1
    2019-09-12 16:17:32.425 INFO  [80443] [main@120] bufferTest OPTIX_VERSION 60000
    2019-09-12 16:17:32.437 INFO  [80443] [OContext::InitRTX@269]  --rtx 0 setting  OFF
    2019-09-12 16:17:32.504 INFO  [80443] [OContext::CheckDevices@204] 
    Device 0                Quadro RTX 8000 ordinal 0 Compute Support: 7 5 Total Memory: 50958893056

    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: ptxas application ptx input, line 49; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 53; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 55; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 57; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 59; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 61; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 63; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 65; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 67; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 69; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 212; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 214; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 224; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 225; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 227; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 228; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 230; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 231; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 233; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 234; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 236; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 237; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas fatal   : Ptx assembly aborted due to errors returned (218): Invalid ptx)
    Aborted (core dumped)
    (base) [blyth@gilda03 ~]$ 



Works?::

    Y UseOptiX            
    Y UseOptiXBuffer
    ? UseOptiXBufferPP
    Y UseOptiXGeometry     
    N UseOptiXGeometryInstancedStandalone
    Y UseOptiXGeometryStandalone
    Y UseOptiXGeometryTriangles
    Y UseOptiXProgram
    Y UseOptiXProgramPP
    ? UseOptiXRap


Found same failure with this one::

    UseOptiXGeometryInstancedStandalone
         running /tmp/blyth/opticks/UseOptiXGeometryInstancedStandalone/bin/UseOptiXGeometryInstancedStandalone
         terminate called after throwing an instance of 'optix::Exception'
         what():  A supported NVIDIA GPU could not be found


    (base) [blyth@gilda03 UseOptiXGeometryInstancedStandalone]$ ./go.sh 
    bdir /tmp/blyth/opticks/UseOptiXGeometryInstancedStandalone/build name UseOptiXGeometryInstancedStandalone prefix /tmp/blyth/opticks/UseOptiXGeometryInstancedStandalone
    === glm-get : nam glm-0.9.9.5 PWD /tmp/blyth/opticks/UseOptiXGeometryInstancedStandalone/externals/glm hpp glm-0.9.9.5/glm/glm/glm.hpp
    ‘glm’ -> ‘glm-0.9.9.5/glm’
    symbolic link for access without version in path
    optix-install-dir : /home/blyth/local/opticks/externals/OptiX
    ./go.sh: line 68: cd: /tmp/blyth/opticks/UseOptiXGeometryInstancedStandalone/build: No such file or directory
    total 6280
    -rw-rw-r--. 1 blyth blyth   23944 Sep 12 16:33 CMakeCache.txt
    drwxrwxr-x. 5 blyth blyth    4096 Sep 12 16:33 CMakeFiles
    -rw-rw-r--. 1 blyth blyth    3565 Sep 12 16:33 cmake_install.cmake
    lrwxrwxrwx. 1 blyth blyth      15 Sep 12 16:42 glm -> glm-0.9.9.5/glm
    drwxrwxr-x. 3 blyth blyth      17 Sep 12 16:32 glm-0.9.9.5
    -rw-rw-r--. 1 blyth blyth 5963606 Sep 12 16:32 glm-0.9.9.5.zip
    -rw-rw-r--. 1 blyth blyth     476 Sep 12 16:33 install_manifest.txt
    -rw-rw-r--. 1 blyth blyth    8109 Sep 12 16:33 Makefile
    -rwxrwxr-x. 1 blyth blyth  380472 Sep 12 16:33 UseOptiXGeometryInstancedStandalone
    -rw-rw-r--. 1 blyth blyth    9304 Sep 12 16:33 UseOptiXGeometryInstancedStandalone_generated_box.cu.ptx
    -rw-rw-r--. 1 blyth blyth   11550 Sep 12 16:33 UseOptiXGeometryInstancedStandalone_generated_rubox.cu.ptx
    -rw-rw-r--. 1 blyth blyth   14304 Sep 12 16:33 UseOptiXGeometryInstancedStandalone_generated_UseOptiXGeometryInstancedStandalone.cu.ptx
    [100%] Built target UseOptiXGeometryInstancedStandalone
    [100%] Built target UseOptiXGeometryInstancedStandalone
    Install the project...
    -- Install configuration: "Debug"
    -- Up-to-date: /tmp/blyth/opticks/UseOptiXGeometryInstancedStandalone/ptx/UseOptiXGeometryInstancedStandalone_generated_UseOptiXGeometryInstancedStandalone.cu.ptx
    -- Up-to-date: /tmp/blyth/opticks/UseOptiXGeometryInstancedStandalone/ptx/UseOptiXGeometryInstancedStandalone_generated_box.cu.ptx
    -- Up-to-date: /tmp/blyth/opticks/UseOptiXGeometryInstancedStandalone/ptx/UseOptiXGeometryInstancedStandalone_generated_rubox.cu.ptx
    -- Up-to-date: /tmp/blyth/opticks/UseOptiXGeometryInstancedStandalone/bin/UseOptiXGeometryInstancedStandalone
    running /tmp/blyth/opticks/UseOptiXGeometryInstancedStandalone/bin/UseOptiXGeometryInstancedStandalone
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: ptxas application ptx input, line 57; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 60; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 62; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 64; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 66; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 68; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 70; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 72; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 74; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 76; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 802; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 804; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 972; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 973; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 975; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 976; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 978; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 979; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 981; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 982; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas application ptx input, line 984; error   : Feature 'activemask' requires PTX ISA .version 6.2 or later
    ptxas application ptx input, line 985; error   : Feature 'shfl.sync' requires PTX ISA .version 6.0 or later
    ptxas fatal   : Ptx assembly aborted due to errors returned (218): Invalid ptx)
    ./go.sh: line 94: 87399 Aborted                 (core dumped) RTX=0 $name
    non-zero RC
    (base) [blyth@gilda03 UseOptiXGeometryInstancedStandalone]$ 


* https://devtalk.nvidia.com/default/topic/1055743/problem-with-turning-off-rtx-mode-on-gtx-1080/



change to 435.21 driver : opticks-t now 0/411 FAILS : even torus tests pass with 435.21 and Quadro RTX 8000
-------------------------------------------------------------------------------------------------------------


While still in no graphics mode, try the example that failed.::

   cd /home/blyth/opticks/examples/UseOptiXGeometryInstancedStandalone
   ./go.sh 


It seems to work with the 435.21 driver.

Even the 2 torus tests that normally fail passed::

    FAILS:  0   / 411   :  Thu Sep 12 18:40:29 2019


BUT checking performance with::

   scan-ph
   an ; ip profilesmryplot.py --cvd 0 --gpu Quadro_RTX_8000

suggests that RTX mode is not making any difference, or not succeeding to be switched on.


hmm : did i add some doubles computation for validation matching improvement (eg he logdouble?)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    (base) [blyth@gilda03 ~]$ oxrap-f64
    ptx.py /home/blyth/local/opticks/installcache/PTX --exclude exception
    ptx.py /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx
     189 : TOTAL .f64 lines in function regions of the PTX 
       0 :  line:0228 : .visible .entry nothing()(  
       0 :  line:0239 : .visible .entry dumpseed()(  
       0 :  line:0318 : .visible .entry trivial()(  
       0 :  line:0423 : .visible .entry zrngtest()(  
       0 :  line:0653 : .visible .entry tracetest()(  
     189 :  line:1487 : .visible .entry generate()(  
       0 :  line:5428 : .visible .entry exception()(  
    ptx.py /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_cbrtTest.cu.ptx
     109 : TOTAL .f64 lines in function regions of the PTX 
     109 :  line:0080 : .visible .entry cbrtTest()(  
       0 :  line:0492 : .visible .entry exception()(  
    ptx.py /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_Roots3And4Test.cu.ptx
     326 : TOTAL .f64 lines in function regions of the PTX 
     326 :  line:0080 : .visible .entry Roots3And4Test()(  
       0 :  line:1151 : .visible .entry exception()(  
    (base) [blyth@gilda03 ~]$ 

::

    (base) [blyth@gilda03 cu]$ opticks-f WITH_LOGDOUBLE
    ./optickscore/OpticksSwitches.h:#define WITH_LOGDOUBLE 1
    ./optickscore/OpticksSwitches.h://#define WITH_LOGDOUBLE_ALT 1
    ./optickscore/OpticksSwitches.h:#ifdef WITH_LOGDOUBLE
    ./optickscore/OpticksSwitches.h:    ss << "WITH_LOGDOUBLE " ;   
    ./optickscore/OpticksSwitches.h:#elif WITH_LOGDOUBLE_ALT
    ./optickscore/OpticksSwitches.h:    ss << "WITH_LOGDOUBLE_ALT " ;   
    ./optixrap/cu/propagate.h:#ifdef WITH_LOGDOUBLE
    ./optixrap/cu/propagate.h:#elif WITH_LOGDOUBLE_ALT
    (base) [blyth@gilda03 opticks]$ 


After comment WITH_LOGDOUBLE and rebuild, reduce from 189 to 87::

    ptx.py /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx
      87 : TOTAL .f64 lines in function regions of the PTX 
       0 :  line:0228 : .visible .entry nothing()(  
       0 :  line:0239 : .visible .entry dumpseed()(  
       0 :  line:0318 : .visible .entry trivial()(  
       0 :  line:0423 : .visible .entry zrngtest()(  
       0 :  line:0653 : .visible .entry tracetest()(  
      87 :  line:1487 : .visible .entry generate()(  
       0 :  line:5241 : .visible .entry exception()(  
    (base) [blyth@gilda03 opticks]$ 



under scan-ph-2 try WITH_LOGDOUBLE commented 
----------------------------------------------


* RTX perf jump is back when cut out those f64, about the same with Quadro_RTX_8000 as with TITAN_RTX 





try geocache-bench360
---------------------------

* shows RTX mode to be giving good speedup with raytrace timings, 

  * x6.5 in triangulated
  * x3.7 analytic  

* absolute timings for analytic with RTX ON and OFF with Quadro 8000 RTX 
  are very close to TITAN RTX times  


::

    (base) [blyth@gilda03 ~]$ geocache-bench360

    ---  GROUPCOMMAND : geocache-bench360   GEOFUNC : - 
    OpSnapTest --envkey --target 62594 --eye 0,0,0 --look 0,0,1 --up 1,0,0 --snapconfig steps=5,eyestartx=0.25,eyestopx=0.25,eyestarty=0.25,eyestopy=0.25,eyestartz=0.25,eyestopz=0.25 --size 10240,5760,1 --enabledmergedmesh 1,2,3,4,5 --cameratype 2 --embedded --cvd 0 --rtx 2 --runfolder geocache-bench360 --runstamp 1568291725 --runlabel R2_Quadro_RTX_8000 
    bench0
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
    /home/blyth/.opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1
                    20190912_203525  launchAVG      rfast      rslow      prelaunch000 
                 R2_Quadro_RTX_8000      0.063      1.000      0.152           2.172  /home/blyth/local/opticks/results/geocache-bench360/R2_Quadro_RT 
                 R1_Quadro_RTX_8000      0.149      2.372      0.360           2.654  /home/blyth/local/opticks/results/geocache-bench360/R1_Quadro_RT 
                 R0_Quadro_RTX_8000      0.415      6.596      1.000           1.642  /home/blyth/local/opticks/results/geocache-bench360/R0_Quadro_RT 

    bench.py --name geocache-bench360


    ---  GROUPCOMMAND : geocache-bench360 --xanalytic  GEOFUNC : - 
    OpSnapTest --envkey --target 62594 --eye 0,0,0 --look 0,0,1 --up 1,0,0 --snapconfig steps=5,eyestartx=0.25,eyestopx=0.25,eyestarty=0.25,eyestopy=0.25,eyestartz=0.25,eyestopz=0.25 --size 10240,5760,1 --enabledmergedmesh 1,2,3,4,5 --cameratype 2 --embedded --cvd 0 --rtx 1 --runfolder geocache-bench360 --runstamp 1568292358 --runlabel R1_Quadro_RTX_8000 --xanalytic 
    bench0
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
    /home/blyth/.opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1
                    20190912_204558  launchAVG      rfast      rslow      prelaunch000 
                 R1_Quadro_RTX_8000      0.213      1.000      0.270           1.965  /home/blyth/local/opticks/results/geocache-bench360/R1_Quadro_RT 
                 R0_Quadro_RTX_8000      0.788      3.705      1.000           1.723  /home/blyth/local/opticks/results/geocache-bench360/R0_Quadro_RT 


     








