Install Opticks onto new machine
=====================================

Install CentOS 7 using DVD and Update Kernel
----------------------------------------------

* magic key F12 to control BIOS
* configure partition for Linux (NB stomped on Windows NTFS: so probable Windows problem)
* install CentOS 7 from DVD
* update the kernel "yum update"
* reboot 


Install as root : CUDA + NVIDIA Driver
--------------------------------------

* use runfile approach to install following https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

* remember to disable Nouveau with "dracut"

Find graphics not working, so follow what did previously (see onvidia-disabling-nouveau)::
  
   yum reinstall `rpm -qa | egrep -i "xorg|mesa"` 

* find that GNOME crashes with glxgears so try with Xfce


Install as user : OptiX 6.0.0
----------------------------------

* OptiX 6 precompiled samples work in Xfce 


SSH setup of new node
--------------------------------

* from another node with env already setup

::

    ssh--putkey C 

* hmm, some problem passwordless not working : fixed by rename authorized_keys2 to authorized_keys


Mercurial, Vim
----------------

::

    sudo yum install mecurial 

From other node, get mercurial + vim config::

    scp .hgrc C:
    scp .vimrc C:


SSH keys setup, bitbucket hookup
---------------------------------

Generate keys (ssh-keygen) and for the new machine 
and upload id_rsa.pub to bitbucket web interface for user blyth 
(NB not adding for only Opticks repo).


Clone Opticks using SSH
-------------------------

::

    cd ; hg clone ssh://hg@bitbucket.org/simoncblyth/opticks


Copy over bash setup
----------------------

::

     scp .opticks_setup C:
     scp .bashrc C:


Check bash setup
----------------------

* https://simoncblyth.bitbucket.io/opticks/docs/opticks.html#check-your-bash-environment-setup

::

   bash -lc "opticks- ; opticks-info "


System CMake is way too old, install 3.14 with ocmake-
---------------------------------------------------------

::

    [blyth@gilda03 ~]$ cmake --version
    cmake version 2.8.12.2

::

    ocmake-
    ocmake-info
    ocmake--



setup sudoer on CentOS 7, follow RHEL instructions
-------------------------------------------------------

* https://developers.redhat.com/blog/2018/08/15/how-to-enable-sudo-on-rhel/

::

   su    # become root
   usermod -aG wheel blyth    # add user blyth to wheel group
   exit

From a fresh session can use sudo::

    [blyth@gilda03 ~]$ sudo touch hello


Install boost, not needed already there
------------------------------------------

::

    [blyth@gilda03 ~]$ sudo yum install boost 
    Loaded plugins: fastestmirror, langpacks
    Loading mirror speeds from cached hostfile
     * base: mirrors.tuna.tsinghua.edu.cn
     * epel: mirrors.tuna.tsinghua.edu.cn
     * extras: mirrors.tuna.tsinghua.edu.cn
     * updates: mirrors.tuna.tsinghua.edu.cn
    Package boost-1.53.0-27.el7.x86_64 already installed and latest version
    Nothing to do
    [blyth@gilda03 ~]$ 
        
        
opticks-externals-install
----------------------------

* from glfw and glew, note same error : opticks-cmake-generator: command not found... 
* fix that in opticks.bash, but guess that "Linux Makefiles" is default anyhow 


* note that opticksdata clone taking long time, 

  TODO: extracate the small part of that which is still needed into a new repo called "opticksgdml" perhaps ?

* ImplicitMesher clone also slow


checking glfw and glew which had errors
------------------------------------------

* the glfw- build was not done, so glfw-wipe and then glfw--
* glew- appears ok


om-install
-------------

* runs into errors with assimprap must have been assimp- problems , maybe from glfw-


Rerun shows that is was not done, there was an opticks-cmake-generator there too::

    assimp-
    assimp--

Resume::

    om-install


Gets to optixrap where FindOptiX fails.

::

    opticks-optix-install-dir
    /home/blyth/local/opticks/externals/OptiX


reinstall OptiX into the right place
-------------------------------------

::

    [blyth@gilda03 ~]$ cd /home/blyth/local/opticks/externals
    [blyth@gilda03 externals]$ sh /home/blyth/NVIDIA-OptiX-SDK-6.0.0-linux64-25650775.sh


After accepting the license respond Y to install with subdir "NVIDIA-OptiX-SDK-6.0.0-linux64"::

    Do you accept the license? [yN]: 
    y
    By default the NVIDIA OptiX will be installed in:
      "/home/blyth/local/opticks/externals/NVIDIA-OptiX-SDK-6.0.0-linux64"
    Do you want to include the subdirectory NVIDIA-OptiX-SDK-6.0.0-linux64?
    Saying no will install in: "/home/blyth/local/opticks/externals" [Yn]: 
    Y

    Using target directory: /home/blyth/local/opticks/externals/NVIDIA-OptiX-SDK-6.0.0-linux64
    Extracting, please wait...

    Unpacking finished successfully


Then plant a symbolic link::

    [blyth@gilda03 externals]$ ln -s NVIDIA-OptiX-SDK-6.0.0-linux64 OptiX

    [blyth@gilda03 ~]$ l ~/local/opticks/externals/OptiX/
    total 16
    drwxrwxr-x. 41 blyth blyth 4096 Sep  9 21:20 SDK
    drwxrwxr-x.  4 blyth blyth 4096 Sep  9 21:20 SDK-precompiled-samples
    drwxrwxr-x.  5 blyth blyth 4096 Sep  9 21:19 include
    drwxrwxr-x.  2 blyth blyth  221 Sep  9 21:19 doc
    drwxrwxr-x.  2 blyth blyth 4096 Sep  9 21:19 lib64
    [blyth@gilda03 ~]$ 


install oxrap now succeeds
-----------------------------

::

   oxrap-
   oxrap-c
   om-install


resume om-install
------------------

::
   
   cd ~/opticks
   om-install


gets thru to oglrap, whence get an imgui related fail
-------------------------------------------------------

Rerun the imgui- install, another opticks-cmake-generator victim::

    imgui-
    imgui--

Now oglrap succeeds::

    oglrap-
    oglrap-c
    om-install


resume om-install again, it completes now
--------------------------------------------

::
   
   cd ~/opticks  # o 
   om-install


opticks-t declines to run
---------------------------------

::

    [blyth@gilda03 opticks]$ opticks-
    [blyth@gilda03 opticks]$ opticks-t
    === opticks-check-installcache : /home/blyth/local/opticks/installcache : missing RNG : curand seeds created by opticks-prepare-installcache cudarap-prepare-installcache
    === opticks-check-installcache : /home/blyth/local/opticks/installcache : missing OKC : GFlags ini files classifying photon source and history states : created by opticks-prepare-installcache OpticksPrepareInstallCache_OKC
    === opticks-t- : ABORT : missing installcache components : create with opticks-prepare-installcache
    [blyth@gilda03 opticks]$ 


opticks-prepare-installcache
------------------------------

::

    [blyth@gilda03 opticks]$ opticks-prepare-installcache
    === opticks-prepare-installcache : generating RNG seeds into installcache
    2019-09-09 21:33:54.219 INFO  [115808] [main@54]  work 3000000 max_blocks 128 seed 0 offset 0 threads_per_block 256 cachedir /home/blyth/local/opticks/installcache/RNG
     init_rng_wrapper sequence_index   0  thread_offset       0  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time     3.1662 ms 
     init_rng_wrapper sequence_index   1  thread_offset   32768  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time     1.4346 ms 
     init_rng_wrapper sequence_index   2  thread_offset   65536  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time     2.0050 ms 
     init_rng_wrapper sequence_index   3  thread_offset   98304  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time     2.5334 ms 
     init_rng_wrapper sequence_index   4  thread_offset  131072  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time     2.8508 ms 
     init_rng_wrapper sequence_index   5  thread_offset  163840  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time     3.3557 ms 

     ...


now opticks-t runs but get 42/411 fails from lack of legacy geocache
----------------------------------------------------------------------

::

    FAILS:  42  / 411   :  Mon Sep  9 21:36:12 2019   
      13 /53  Test #13 : GGeoTest.GScintillatorLibTest                 ***Exception: SegFault         0.12   
      16 /53  Test #16 : GGeoTest.GBndLibTest                          Child aborted***Exception:     0.12   
      17 /53  Test #17 : GGeoTest.GBndLibInitTest                      Child aborted***Exception:     0.63   
      30 /53  Test #30 : GGeoTest.GPtsTest                             Child aborted***Exception:     0.13   
      33 /53  Test #33 : GGeoTest.GPmtTest                             Child aborted***Exception:     0.11   
      34 /53  Test #34 : GGeoTest.BoundariesNPYTest                    Child aborted***Exception:     0.12   
      35 /53  Test #35 : GGeoTest.GAttrSeqTest                         Child aborted***Exception:     0.63   
      39 /53  Test #39 : GGeoTest.GGeoLibTest                          Child aborted***Exception:     0.13   
      40 /53  Test #40 : GGeoTest.GGeoTest                             Child aborted***Exception:     0.12   
      41 /53  Test #41 : GGeoTest.GMakerTest                           Child aborted***Exception:     0.13   
      48 /53  Test #48 : GGeoTest.GSurfaceLibTest                      Child aborted***Exception:     0.11   
      50 /53  Test #50 : GGeoTest.NLookupTest                          Child aborted***Exception:     0.17   
      51 /53  Test #51 : GGeoTest.RecordsNPYTest                       Child aborted***Exception:     0.12   
      52 /53  Test #52 : GGeoTest.GSceneTest                           Child aborted***Exception:     0.13   
      1  /3   Test #1  : OpticksGeoTest.OpticksGeoTest                 Child aborted***Exception:     0.16   
      2  /3   Test #2  : OpticksGeoTest.OpticksHubTest                 Child aborted***Exception:     0.15   
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     2.63   
      7  /24  Test #7  : OptiXRapTest.boundaryTest                     Child aborted***Exception:     0.36   
      8  /24  Test #8  : OptiXRapTest.boundaryLookupTest               Child aborted***Exception:     0.37   
      12 /24  Test #12 : OptiXRapTest.rayleighTest                     Child aborted***Exception:     0.31   
      17 /24  Test #17 : OptiXRapTest.eventTest                        Child aborted***Exception:     0.37   
      18 /24  Test #18 : OptiXRapTest.interpolationTest                Child aborted***Exception:     0.33   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.94   
      1  /5   Test #1  : OKOPTest.OpIndexerTest                        Child aborted***Exception:     0.35   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     0.34   
      5  /5   Test #5  : OKOPTest.OpSnapTest                           Child aborted***Exception:     0.34   
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     0.31   
      3  /5   Test #3  : OKTest.OTracerTest                            Child aborted***Exception:     0.35   
      1  /34  Test #1  : CFG4Test.CMaterialLibTest                     Child aborted***Exception:     0.41   
      2  /34  Test #2  : CFG4Test.CMaterialTest                        Child aborted***Exception:     0.43   
      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     0.42   
      5  /34  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     0.40   
      6  /34  Test #6  : CFG4Test.CGeometryTest                        Child aborted***Exception:     0.41   
      7  /34  Test #7  : CFG4Test.CG4Test                              Child aborted***Exception:     0.42   
      22 /34  Test #22 : CFG4Test.CGenstepCollectorTest                Child aborted***Exception:     0.38   
      23 /34  Test #23 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     0.42   
      25 /34  Test #25 : CFG4Test.CGROUPVELTest                        Child aborted***Exception:     0.42   
      29 /34  Test #29 : CFG4Test.CRandomEngineTest                    Child aborted***Exception:     0.37   
      32 /34  Test #32 : CFG4Test.CCerenkovGeneratorTest               Child aborted***Exception:     0.42   
      33 /34  Test #33 : CFG4Test.CGenstepSourceTest                   Child aborted***Exception:     0.40   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     0.43   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      0.77   
    [blyth@gilda03 opticks]$ 



run legacy geocache prep
--------------------------------

::

    [blyth@gilda03 opticks]$ op.sh -G


* completes in a few seconds only 


check OKTest graphically
-------------------------

* get API error from CUDA as OPTICKS_DEFAULT_INTEROP_CVD envvar is set to 1 (from multi GPU workstation)
  in ~/opticks_setup and this node has only one GPU 

* change the value to 0

* now OKTest works and pops up window show DayaBay Near geometry with a propagation, 
  if you know the keys to use (Q, A, O, G) 


try again opticks-t, down to 7/411 fails
------------------------------------------

::

    FAILS:  7   / 411   :  Mon Sep  9 21:40:15 2019   
      30 /53  Test #30 : GGeoTest.GPtsTest                             Child aborted***Exception:     0.14   
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     2.38   
      18 /24  Test #18 : OptiXRapTest.interpolationTest                ***Failed                      2.31   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.95   
      7  /34  Test #7  : CFG4Test.CG4Test                              ***Exception: SegFault         13.22  
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: SegFault         18.09  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      0.77   
    [blyth@gilda03 opticks]$ 


These are from missing the direct mode JUNO geocache. So create it::

    geocache-
    geocache-create

Runs this from terminal on the node itself as it pops up a window with JUNO geometry (no propagation).
Switching from rasterized to raytrace works with O key. 


try again opticks-t, still 6/411 fails : only GPtsTest was fixed 
------------------------------------------------------------------

::

    FAILS:  6   / 411   :  Mon Sep  9 21:58:45 2019   
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     2.39   
      18 /24  Test #18 : OptiXRapTest.interpolationTest                ***Failed                      2.63   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.99   
      7  /34  Test #7  : CFG4Test.CG4Test                              ***Exception: SegFault         13.28  
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: SegFault         18.40  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      0.79   
    [blyth@gilda03 opticks]$ 


interpolationTest + IntegrationTests.tboolean.box? from lack of NumPy for analysis ? Need to install anaconda
------------------------------------------------------------------------------------------------------------------

::

    [blyth@gilda03 opticks]$ interpolationTest
    2019-09-09 22:00:43.390 INFO  [130748] [Opticks::init@339] COMPUTE_MODE forced_compute 
    ...
    2019-09-09 22:00:45.129 INFO  [130748] [interpolationTest::ana@178]  path /home/blyth/opticks/optixrap/tests/interpolationTest_interpol.py
    Traceback (most recent call last):
      File "/home/blyth/opticks/optixrap/tests/interpolationTest_interpol.py", line 22, in <module>
        import os,sys, numpy as np, logging
    ImportError: No module named numpy
    2019-09-09 22:00:45.499 INFO  [130748] [SSys::run@91] python /home/blyth/opticks/optixrap/tests/interpolationTest_interpol.py rc_raw : 256 rc : 1
    2019-09-09 22:00:45.499 ERROR [130748] [SSys::run@98] FAILED with  cmd python /home/blyth/opticks/optixrap/tests/interpolationTest_interpol.py RC 1
    2019-09-09 22:00:45.499 INFO  [130748] [interpolationTest::ana@180]  RC 1




CG4Test + OKG4Test fails, from lack of m_engine when not aligned : FIXED this
-------------------------------------------------------------------------------


setup env 
-----------

::

    cd ; hg clone ssh://hg@bitbucket.org/simoncblyth/env

* the hookup already in the copyied over bash setup 


Setup for google with Firefox
-----------------------------

Copy the socks machinery + ssh config for remote node B::

   [blyth@localhost ~]$ scp -r home C:
   [blyth@localhost ~]$ scp gfw.pac C:


1. Setup passwordless from C to B, with "ssh--putkey B", start agent "sas", check "ssh B".
2. Start proxy, socks-- 
3. configure Firefox 
 
   * Automatic proxy configuration URL : file:///home/blyth/gfw.pac
   * proxt DNS

4. check google with a Firefox search, works


Kick off anaconda download : 476MB will take 2hrs
-----------------------------------------------------


Before NumPy in place, check opticks-t : stand at 4/411 fails 
--------------------------------------------------------------

* 2 are known issues for OptiX 6.0.0 and Torus
* other 2 are expected to be fixed with NumPy in place

::

    FAILS:  4   / 411   :  Mon Sep  9 22:44:54 2019   
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   ***Exception: Other            2.39   
      18 /24  Test #18 : OptiXRapTest.interpolationTest                ***Failed                      2.32   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest ***Exception: Other            2.95   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      0.78   
    [blyth@gilda03 cfg4]$ 


opticks-t down to 2/411 fails after Anaconda2 installed for NumPy
-------------------------------------------------------------------

* after Anaconda2 done, down to 

