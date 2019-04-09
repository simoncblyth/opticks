nvidia-driver-fails-following-yum-update
============================================

See Also
----------

See onvidia-vi for how the driver was updated to fix this.


After the reboot
-------------------

Reboot after a "yum update" has lost some GPU driver setup, 

* screen shows in low resolution and uname reports a new kernel version. 
* nvidia-smi complains of no driver
* opticks-t has lots of fails

::

    [blyth@localhost issues]$ uname -a
    Linux localhost.localdomain 3.10.0-957.10.1.el7.x86_64 #1 SMP Mon Mar 18 15:06:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux

    [blyth@localhost issues]$ nvidia-smi
    NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.


opticks-t has lots of fails::


    LS:
      2  /5   Test #2  : CUDARapTest.cuRANDWrapperTest                 ***Failed                      0.08   
      1  /15  Test #1  : ThrustRapTest.CBufSpecTest                    Child aborted***Exception:     0.25   
      2  /15  Test #2  : ThrustRapTest.TBufTest                        Child aborted***Exception:     0.21   
      3  /15  Test #3  : ThrustRapTest.TRngBufTest                     Child aborted***Exception:     0.62   
      4  /15  Test #4  : ThrustRapTest.expandTest                      Child aborted***Exception:     0.20   
      5  /15  Test #5  : ThrustRapTest.iexpandTest                     Child aborted***Exception:     0.19   
      6  /15  Test #6  : ThrustRapTest.issue628Test                    Child aborted***Exception:     0.20   
      7  /15  Test #7  : ThrustRapTest.printfTest                      Child aborted***Exception:     0.20   
      8  /15  Test #8  : ThrustRapTest.repeated_rangeTest              Child aborted***Exception:     0.21   
      9  /15  Test #9  : ThrustRapTest.strided_rangeTest               Child aborted***Exception:     0.21   
      10 /15  Test #10 : ThrustRapTest.strided_repeated_rangeTest      Child aborted***Exception:     0.22   
      11 /15  Test #11 : ThrustRapTest.float2intTest                   Child aborted***Exception:     0.20   
      12 /15  Test #12 : ThrustRapTest.thrust_curand_estimate_pi       Child aborted***Exception:     0.22   
      13 /15  Test #13 : ThrustRapTest.thrust_curand_printf            Child aborted***Exception:     0.19   
      14 /15  Test #14 : ThrustRapTest.thrust_curand_printf_redirect   Child aborted***Exception:     0.21   
      15 /15  Test #15 : ThrustRapTest.thrust_curand_printf_redirect2  Child aborted***Exception:     0.22   
      1  /18  Test #1  : OptiXRapTest.OPropertyLibTest                 Child aborted***Exception:     0.43   
      2  /18  Test #2  : OptiXRapTest.OScintillatorLibTest             Child aborted***Exception:     0.29   
      3  /18  Test #3  : OptiXRapTest.OOTextureTest                    Child aborted***Exception:     0.27   
      4  /18  Test #4  : OptiXRapTest.OOMinimalTest                    Child aborted***Exception:     0.27   
      5  /18  Test #5  : OptiXRapTest.OOMinimalRedirectTest            Child aborted***Exception:     0.28   
      6  /18  Test #6  : OptiXRapTest.OOContextTest                    Child aborted***Exception:     0.27   
      7  /18  Test #7  : OptiXRapTest.OOContextUploadDownloadTest      Child aborted***Exception:     0.29   
      8  /18  Test #8  : OptiXRapTest.LTOOContextUploadDownloadTest    Child aborted***Exception:     0.27   
      9  /18  Test #9  : OptiXRapTest.OOboundaryTest                   Child aborted***Exception:     0.28   
      10 /18  Test #10 : OptiXRapTest.OOboundaryLookupTest             Child aborted***Exception:     0.31   
      11 /18  Test #11 : OptiXRapTest.OOtex0Test                       Child aborted***Exception:     0.26   
      12 /18  Test #12 : OptiXRapTest.OOtexTest                        Child aborted***Exception:     0.26   
      13 /18  Test #13 : OptiXRapTest.bufferTest                       Child aborted***Exception:     0.28   
      14 /18  Test #14 : OptiXRapTest.OEventTest                       Child aborted***Exception:     0.53   
      15 /18  Test #15 : OptiXRapTest.OInterpolationTest               Child aborted***Exception:     0.57   
      16 /18  Test #16 : OptiXRapTest.ORayleighTest                    Child aborted***Exception:     0.55   
      17 /18  Test #17 : OptiXRapTest.intersect_analytic_test          Child aborted***Exception:     0.26   
      18 /18  Test #18 : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     0.27   
      1  /5   Test #1  : OKOPTest.OpIndexerTest                        Child aborted***Exception:     0.58   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     0.55   
      3  /5   Test #3  : OKOPTest.dirtyBufferTest                      Child aborted***Exception:     0.30   
      4  /5   Test #4  : OKOPTest.compactionTest                       Child aborted***Exception:     0.29   
      5  /5   Test #5  : OKOPTest.OpSnapTest                           Child aborted***Exception:     0.56   
      2  /5   Test #2  : OKTest.OKTest                                 ***Failed                      0.33   
      3  /5   Test #3  : OKTest.OTracerTest                            ***Failed                      0.34   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Failed                      0.99   



Rebooting again and picking the second from list 
--------------------------------------------------

* back to normal resolution, and uname is reporting a different kernel version ?
* nvidia-smi working again and 

::

    [blyth@localhost ~]$ uname -a
    Linux localhost.localdomain 3.10.0-862.6.3.el7.x86_64 #1 SMP Tue Jun 26 16:32:21 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux

    [blyth@localhost ~]$ cat /etc/redhat-release 
    CentOS Linux release 7.6.1810 (Core) 


    [blyth@localhost ~]$ nvidia-smi
    Sun Apr  7 23:26:15 2019       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 396.26                 Driver Version: 396.26                    |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  TITAN V             Off  | 00000000:73:00.0  On |                  N/A |
    | 28%   41C    P2    29W / 250W |     87MiB / 12066MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0     13620      G   /usr/bin/X                                    86MiB |
    +-----------------------------------------------------------------------------+
    [blyth@localhost ~]$ 



Incidentally notice that graphics accelearation appears lost 
during the running of OOMinimalRedirectTest::

     .    Start  4: OptiXRapTest.OOMinimalTest
     4/18 Test  #4: OptiXRapTest.OOMinimalTest ...................   Passed   28.00 sec
          Start  5: OptiXRapTest.OOMinimalRedirectTest
     5/18 Test  #5: OptiXRapTest.OOMinimalRedirectTest ...........   Passed   27.91 sec


Also three fails all from same cause::

    FAILS:
      2  /5   Test #2  : OKTest.OKTest                                 ***Failed                      0.34   
      3  /5   Test #3  : OKTest.OTracerTest                            ***Failed                      0.33   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Failed                      1.03   



::

    OKTest
    ...
    2019-04-07 23:33:34.302 ERROR [22092] [OpticksViz::init@135] renderMode (null)
    2019-04-07 23:33:34.302 INFO  [22092] [OpticksViz::setupRendermode@233] OpticksViz::setupRendermode [-]
    2019-04-07 23:33:34.302 INFO  [22092] [OpticksViz::setupRendermode@248] OpticksViz::setupRendermode rmode axis,genstep,nopstep,photon,record,
    2019-04-07 23:33:34.305 INFO  [22092] [Scene::setRecordStyle@1127] point
    GLX: Failed to create context: GLXBadFBConfig[blyth@localhost tests]$ 




::

   GLX: Failed to create context: GLXBadFBConfig


Also same::

    AxisAppCheck 

  


::

    blyth@localhost issues]$ xrandr
    Screen 0: minimum 8 x 8, current 2560 x 1440, maximum 32767 x 32767
    DP-0 disconnected (normal left inverted right x axis y axis)
    DP-1 disconnected (normal left inverted right x axis y axis)
    DP-2 connected primary 2560x1440+0+0 (normal left inverted right x axis y axis) 600mm x 340mm
       2560x1440     59.95*+ 144.00   120.00    99.95    84.98    23.97  
       1024x768      60.00  
       800x600       60.32  
       640x480       59.94  
    DP-3 disconnected (normal left inverted right x axis y axis)
    HDMI-0 disconnected (normal left inverted right x axis y axis)
    DP-4 disconnected (normal left inverted right x axis y axis)
    DP-5 disconnected (normal left inverted right x axis y axis)
    [blyth@localhost issues]$ 
    [blyth@localhost issues]$ uname -a
    Linux localhost.localdomain 3.10.0-862.6.3.el7.x86_64 #1 SMP Tue Jun 26 16:32:21 UTC 2018 x86_64 x86_64 x86_64 GNU/Linux
    [blyth@localhost issues]$



