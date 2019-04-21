multi_gpu_opticks_interop_mode_requires_one_visible_device
=============================================================

Overview
----------

When doing interop dev, simply use TITAN RTX (which graphics is using) with 

    CUDA_VISIBLE_DEVICES=1 OKTest ... 


opticks-t with TITAN V and TITAN RTX
---------------------------------------------

With CUDA_VISIBLE_DEVICES unset or set to 0,1 get 8 additional fails 
in addition to the 2 know quartic problems.::

    FAILS:
      9  /19  Test #9  : OptiXRapTest.LTOOContextUploadDownloadTest    Child aborted***Exception:     1.22   
      14 /19  Test #14 : OptiXRapTest.bufferTest                       Child aborted***Exception:     1.33   
      15 /19  Test #15 : OptiXRapTest.OEventTest                       Child aborted***Exception:     1.62   

      18 /19  Test #18 : OptiXRapTest.intersect_analytic_test          Child aborted***Exception:     1.34   
      19 /19  Test #19 : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.36   
                           known quartic problem 

      2  /5   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     3.92   
      3  /5   Test #3  : OKOPTest.dirtyBufferTest                      Child aborted***Exception:     1.35   
      4  /5   Test #4  : OKOPTest.compactionTest                       Child aborted***Exception:     1.36   
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     6.79   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     21.02  
    [blyth@localhost opticks]$ 


Picking TITAN RTX are down to the known two quartic fails::

    [blyth@localhost okg4]$ CUDA_VISIBLE_DEVICES=1 opticks-t
    === om-test-one : okconf          /home/blyth/opticks/okconf                                   /home/blyth/local/opticks/build/okconf                       
    Sun Apr 21 13:15:45 CST 2019
    ...
    FAILS:
      18 /19  Test #18 : OptiXRapTest.intersect_analytic_test          Child aborted***Exception:     1.29   
      19 /19  Test #19 : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     1.19   
    [blyth@localhost build]$ 


OKTest 
-------

With CUDA_VISIBLE_DEVICES unset or "0,1" and the OpenGL context being handled by TITAN RTX
(because thats the one connected to the display!) OKTest fails with *Cannot get device pointers from non-enabled device*::


    [blyth@localhost ~]$ CUDA_VISIBLE_DEVICES=0,1 gdb OKTest
    ...
    2019-04-19 19:35:43.581 ERROR [125184] [Frame::initContext@356] Frame::gl_init_window Renderer: TITAN RTX/PCIe/SSE2
    2019-04-19 19:35:43.581 ERROR [125184] [Frame::initContext@357] Frame::gl_init_window OpenGL version supported 4.6.0 NVIDIA 418.56

    2019-04-19 19:35:45.376 INFO  [125184] [OEvent::uploadGensteps@311] OEvent::uploadGensteps (INTEROP) SKIP OpenGL BufferId 59
    2019-04-19 19:35:45.376 INFO  [125184] [OpEngine::propagate@117] [
    2019-04-19 19:35:45.376 INFO  [125184] [OpSeeder::seedComputeSeedsFromInteropGensteps@63] OpSeeder::seedComputeSeedsFromInteropGensteps : WITH_SEED_BUFFER 
    OFormat::ElementSizeInBytes UNSIGNED_INT : 4
    terminate called after throwing an instance of 'optix::Exception'
      what():  Invalid value (Details: Function "RTresult _rtBufferGetDevicePointer(RTbuffer, int, void**)" caught exception: Cannot get device pointers from non-enabled device.)

    Program received signal SIGABRT, Aborted.
    (gdb) bt
    #0  0x00007fffeafc6207 in raise () from /lib64/libc.so.6
    #1  0x00007fffeafc78f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffeb8d57d5 in __gnu_cxx::__verbose_terminate_handler() () from /lib64/libstdc++.so.6
    #3  0x00007fffeb8d3746 in ?? () from /lib64/libstdc++.so.6
    #4  0x00007fffeb8d3773 in std::terminate() () from /lib64/libstdc++.so.6
    #5  0x00007fffeb8d3993 in __cxa_throw () from /lib64/libstdc++.so.6
    #6  0x00007ffff673c254 in optix::APIObj::checkError (this=0x9d87b20, code=RT_ERROR_INVALID_VALUE) at /usr/local/OptiX_600/include/optixu/optixpp_namespace.h:2151
    #7  0x00007ffff677c4b9 in OBufBase::getDevicePtr() () from /home/blyth/local/opticks/lib64/libOptiXRap.so
    #8  0x00007ffff677c68e in OBufBase::bufspec() () from /home/blyth/local/opticks/lib64/libOptiXRap.so
    #9  0x00007ffff6aa5e14 in OpSeeder::seedComputeSeedsFromInteropGensteps (this=0x832b760) at /home/blyth/opticks/okop/OpSeeder.cc:72
    #10 0x00007ffff6aa5c9a in OpSeeder::seedPhotonsFromGensteps (this=0x832b760) at /home/blyth/opticks/okop/OpSeeder.cc:51
    #11 0x00007ffff6aaf93e in OpEngine::propagate (this=0x371ea60) at /home/blyth/opticks/okop/OpEngine.cc:120
    #12 0x00007ffff7bd69e1 in OKPropagator::propagate (this=0x3726a00) at /home/blyth/opticks/ok/OKPropagator.cc:76
    #13 0x00007ffff7bd5ce5 in OKMgr::propagate (this=0x7fffffffd920) at /home/blyth/opticks/ok/OKMgr.cc:102
    #14 0x0000000000402e9a in main (argc=1, argv=0x7fffffffda98) at /home/blyth/opticks/ok/tests/OKTest.cc:14
    (gdb) 


    [blyth@localhost ~]$ CUDA_VISIBLE_DEVICES=0,1 UseOptiX
    OptiX 6.0.0
    Number of Devices = 2

    Device 0: TITAN V
      Compute Support: 7 0
      Total Memory: 12621381632 bytes
    Device 1: TITAN RTX
      Compute Support: 7 5
      Total Memory: 25364987904 bytes
     RT_FORMAT_FLOAT4 size 16

    [blyth@localhost ~]$ CUDA_VISIBLE_DEVICES=1,0 UseOptiX   ## surprised to see that the order does matter
    OptiX 6.0.0
    Number of Devices = 2

    Device 0: TITAN RTX
      Compute Support: 7 5
      Total Memory: 25364987904 bytes
    Device 1: TITAN V
      Compute Support: 7 0
      Total Memory: 12621381632 bytes
     RT_FORMAT_FLOAT4 size 16

    [blyth@localhost ~]$ CUDA_VISIBLE_DEVICES=1 UseOptiX
    OptiX 6.0.0
    Number of Devices = 1

    Device 0: TITAN RTX
      Compute Support: 7 5
      Total Memory: 25364987904 bytes
     RT_FORMAT_FLOAT4 size 16

    [blyth@localhost ~]$ CUDA_VISIBLE_DEVICES=0 UseOptiX
    OptiX 6.0.0
    Number of Devices = 1

    Device 0: TITAN V
      Compute Support: 7 0
      Total Memory: 12621381632 bytes
     RT_FORMAT_FLOAT4 size 16
    [blyth@localhost ~]$ 



As shown above With "CUDA_VISIBLE_DEVICES=0,1" or unset OKTest fails with::

     Cannot get device pointers from non-enabled device

With "CUDA_VISIBLE_DEVICES=1,0" TITAN RTX is listed first and as that matches the GPU OpenGL context it works::

    [blyth@localhost ~]$ CUDA_VISIBLE_DEVICES=1,0 gdb OKTest

With "CUDA_VISIBLE_DEVICES=1" succceeds to run and to raytrace::

    [blyth@localhost ~]$ CUDA_VISIBLE_DEVICES=1 gdb OKTest

With "CUDA_VISIBLE_DEVICES=0", fails with cudaErrorInvalidDevice::

    [blyth@localhost ~]$ CUDA_VISIBLE_DEVICES=0 gdb OKTest
    ...
    2019-04-19 19:26:59.671 INFO  [111166] [OEvent::uploadGensteps@311] OEvent::uploadGensteps (INTEROP) SKIP OpenGL BufferId 59
    2019-04-19 19:26:59.671 INFO  [111166] [OpEngine::propagate@117] [
    2019-04-19 19:26:59.671 INFO  [111166] [OpSeeder::seedComputeSeedsFromInteropGensteps@63] OpSeeder::seedComputeSeedsFromInteropGensteps : WITH_SEED_BUFFER 
    OFormat::ElementSizeInBytes UNSIGNED_INT : 4
    CUDA error at /home/blyth/opticks/cudarap/CResource_.cu:43 code=101(cudaErrorInvalidDevice) "cudaGraphicsGLRegisterBuffer(&resource, buffer_id, flags)" 
    [Thread 0x7fff027fc700 (LWP 111296) exited]


CUDA with multiple GPUs
-------------------------

* https://stackoverflow.com/questions/13781738/how-does-cuda-assign-device-ids-to-gpus

When a computer has multiple CUDA-capable GPUs, each GPU is assigned a device
ID. By default, CUDA kernels execute on device ID 0. You can use
cudaSetDevice(int device) to select a different device.


export CUDA_DEVICE_ORDER=PCI_BUS_ID
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://groups.google.com/forum/#!topic/slurm-users/Fv2cgq80GmU


Docker GPU control
~~~~~~~~~~~~~~~~~~~~

* https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#i-have-multiple-gpu-devices-how-can-i-isolate-them-between-my-containers


How to configure OpenGL to use a particular GPU ?
-----------------------------------------------------

::

    [blyth@localhost ~]$ glxinfo | grep NVIDIA
    server glx vendor string: NVIDIA Corporation
    client glx vendor string: NVIDIA Corporation
    OpenGL vendor string: NVIDIA Corporation
    OpenGL core profile version string: 4.6.0 NVIDIA 418.56
    OpenGL core profile shading language version string: 4.60 NVIDIA
    OpenGL version string: 4.6.0 NVIDIA 418.56
    OpenGL shading language version string: 4.60 NVIDIA
    OpenGL ES profile version string: OpenGL ES 3.2 NVIDIA 418.56
    [blyth@localhost ~]$ 



nvidia-smi
-------------


::

    [root@localhost issues]# nvidia-smi -a

    ==============NVSMI LOG==============

    Timestamp                           : Fri Apr 19 20:12:54 2019
    Driver Version                      : 418.56
    CUDA Version                        : 10.1

    Attached GPUs                       : 2
    GPU 00000000:73:00.0
        Product Name                    : TITAN RTX
        Product Brand                   : Titan
        Display Mode                    : Enabled
        Display Active                  : Enabled
        Persistence Mode                : Disabled
        Accounting Mode                 : Disabled
        Accounting Mode Buffer Size     : 4000
        Driver Model
            Current                     : N/A
            Pending                     : N/A
        Serial Number                   : 0320219051149
        GPU UUID                        : GPU-9b4994a5-1105-10a5-079a-57a494c351bc
        Minor Number                    : 0
    ...
    GPU 00000000:A6:00.0
        Product Name                    : TITAN V
        Product Brand                   : Titan
        Display Mode                    : Disabled
        Display Active                  : Disabled
        Persistence Mode                : Disabled
        Accounting Mode                 : Disabled
        Accounting Mode Buffer Size     : 4000
        Driver Model
            Current                     : N/A
            Pending                     : N/A
        Serial Number                   : 0324917182697
        GPU UUID                        : GPU-50208d32-6612-fcb5-ea38-28ef96349934
        Minor Number                    : 1



Seems no GUI for nvidia settings on CentOS 7
-----------------------------------------------

::

    root@localhost issues]# which nvidia-settings
    /bin/nvidia-settings
    [root@localhost issues]# which nvidia-xconfig
    /bin/nvidia-xconfig
    [root@localhost issues]# ll /bin/nvidia*
    -rwxr-xr-x. 1 root root  27814 Apr 14 15:06 /bin/nvidia-bug-report.sh
    -rwxr-xr-x. 1 root root  76888 Apr 14 15:06 /bin/nvidia-cuda-mps-control
    -rwxr-xr-x. 1 root root  54024 Apr 14 15:06 /bin/nvidia-cuda-mps-server
    -rwxr-xr-x. 1 root root 247760 Apr 14 15:06 /bin/nvidia-debugdump
    -rwxr-xr-x. 1 root root 327312 Apr 14 15:06 /bin/nvidia-installer
    -rwsr-xr-x. 1 root root  33936 Apr 14 15:06 /bin/nvidia-modprobe
    -rwxr-xr-x. 1 root root  48096 Apr 14 15:06 /bin/nvidia-persistenced
    -rwxr-xr-x. 1 root root 299336 Apr 14 15:06 /bin/nvidia-settings
    -rwxr-xr-x. 1 root root 528288 Apr 14 15:06 /bin/nvidia-smi
    lrwxrwxrwx. 1 root root     16 Apr 14 15:06 /bin/nvidia-uninstall -> nvidia-installer
    -rwxr-xr-x. 1 root root 184888 Apr 14 15:06 /bin/nvidia-xconfig
    [root@localhost issues]# 


::

    [root@localhost issues]# nvidia-xconfig

    Using X configuration file: "/etc/X11/xorg.conf".
    Backed up file '/etc/X11/xorg.conf' as '/etc/X11/xorg.conf.backup'
    New X configuration file written to '/etc/X11/xorg.conf'

    [root@localhost issues]# diff /etc/X11/xorg.conf.backup /etc/X11/xorg.conf
    [root@localhost issues]# 


multi GPU ref 
---------------

* :google:`nvidia-xconfig Device0 Device1`

* https://nvidia.custhelp.com/app/answers/detail/a_id/3029/~/using-cuda-and-x


lspci : lists the BusID of the GPUs in hex
------------------------------------------------

::

    root@localhost issues]# lspci | grep VGA
    73:00.0 VGA compatible controller: NVIDIA Corporation TU102 (rev a1)
    a6:00.0 VGA compatible controller: NVIDIA Corporation GV100 [TITAN V] (rev a1)



Titan RTX
------------

* https://www.techpowerup.com/gpu-specs/titan-rtx.c3311

Selecting GPU used for display with BusID key in Device section, converted to decimal
------------------------------------------------------------------------------------------

/etc/X11/xorg.conf::

    ...

    Section "Monitor"
        Identifier     "Monitor0"
        VendorName     "Unknown"
        ModelName      "Unknown"
        HorizSync       28.0 - 33.0
        VertRefresh     43.0 - 72.0
        Option         "DPMS"
    EndSection

    Section "Device"
        Identifier     "Device0"
        Driver         "nvidia"
        VendorName     "NVIDIA Corporation"
    EndSection

    Section "Screen"
        Identifier     "Screen0"
        Device         "Device0"
        Monitor        "Monitor0"
        DefaultDepth    24
        SubSection     "Display"
            Depth       24
        EndSubSection
    EndSection

* https://stackoverflow.com/questions/18382271/how-can-i-modify-xorg-conf-file-to-force-x-server-to-run-on-a-specific-gpu-i-a

Add BusID key, where the value comes from nvidia-smi -a::

    Section "Device"
        Identifier     "Device0"
        Driver         "nvidia"
        VendorName     "NVIDIA Corporation"
        BusID          "PCI:2:0:0"
    EndSection

Reboot and ensure display cable attached to the appropriate GPU.

One very important note; if there are many GPUs installed you will get hex
values from lspci or nvidia-smi like 0000:0A:00.0. You have to either convert
it to decimal like this 10:00:0 or skip leading zero(s) like this A:00:0
(notice 0A is now just A). Credit goes to ossifrage at #ethereum-mining 



* https://devtalk.nvidia.com/default/topic/769851/multi-nvidia-gpus-and-xorg-conf-how-to-account-for-pci-bus-busid-change-/

  Caution BusID may change between reboots






