macos-dtruss-monitor-file-opens
=================================

On Linux can monitor file creates with::

    o.sh --oktest --strace --production --nosave 

Which does the below using strace to log the "open" system call and then a script to filter showing just creations::

    strace -o /tmp/strace.log -e open /home/blyth/local/opticks/lib/OKTest --oktest --strace --production --nosave

    /home/blyth/local/opticks/bin/strace.py -f O_CREAT       
    strace.py -f O_CREAT
     OKTest.log                                                                       :          O_WRONLY|O_CREAT :  0644 
     /var/tmp/blyth/OptiXCache/cache.db                                               :            O_RDWR|O_CREAT :  0666 
     /var/tmp/blyth/OptiXCache/cache.db                                               : O_WRONLY|O_CREAT|O_APPEND :  0666 
     /var/tmp/blyth/OptiXCache/cache.db-wal                                           :            O_RDWR|O_CREAT :  0664 
     /var/tmp/blyth/OptiXCache/cache.db-shm                                           :            O_RDWR|O_CREAT :  0664 
     /home/blyth/.opticks/runcache/CDevice.bin                                        :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/Time.ini                              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/DeltaTime.ini                         :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/VM.ini                                :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/DeltaVM.ini                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/OpticksProfile.npy                    :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/OpticksProfileLabels.npy              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/OpticksProfileAcc.npy                 :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/OpticksProfileAccLabels.npy           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/OpticksProfileLis.npy                 :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/OpticksProfileLisLabels.npy           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/opticks/OKTest/evt/g4live/torch/0/parameters.json                     :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
    /home/blyth/local/opticks/bin/o.sh : RC : 0


macOS does not have strace so try dtruss

1. have to run dtruss via sudo, so create a self contained OKTest_macOS_standalone.sh for use from installed location  
2. cannot use /bin/bash due to SIP protections so switch to /opt/local/bin/bash  
3. works to some extent : but getting SIP denials for dtrace


::

    epsilon:bin blyth$ sudo dtruss -t open /usr/local/opticks/bin/OKTest.sh
    Password:
    dtrace: system integrity protection is on, some features will not be available

    dtrace: failed to execute /usr/local/opticks/bin/OKTest.sh: dtrace cannot control executables signed with restricted entitlements
    epsilon:bin blyth$ which bash
    /opt/local/bin/bash


::

    sudo dtruss -f -t open /opt/local/bin/bash /usr/local/opticks/bin/OKTest.sh


Works to some extent, but getting much less than with strace on Linux presumably futher macOS protections::

    2020-11-24 11:08:06.061 INFO  [12636786] [OpticksHub::loadGeometry@281] [ /usr/local/opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1
    dtrace: 27063 dynamic variable drops with non-empty dirty list
    85455/0xc0d23a:  open("/dev/dtracehelper\0", 0x2, 0xFFFFFFFFE62689F0)		 = 3 0
    85455/0xc0d23a:  open("/dev/tty\0", 0x6, 0xFFFFFFFFE6269900)		 = 3 0
    85455/0xc0d23a:  open("/usr/local/opticks/bin/OKTest.sh\0", 0x0, 0xC)		 = 3 0
    85456/0xc0d244:  fork()		 = 0 0
    85457/0xc0d245:  fork()		 = 0 0
    dtrace: error on enabled probe ID 161 (ID 896: syscall::thread_selfid:entry): invalid user access in action #2 at DIF offset 0
    dtrace: error on enabled probe ID 2170 (ID 163: syscall::open:return): invalid user access in action #5 at DIF offset 0
    dtrace: error on enabled probe ID 161 (ID 896: syscall::thread_selfid:entry): invalid user access in action #2 at DIF offset 0
    dtrace: error on enabled probe ID 2170 (ID 163: syscall::open:return): invalid user access in action #5 at DIF offset 0
    85458/0xc0d248:  fork()		 = 0 0
    dtrace: error on enabled probe ID 2170 (ID 163: syscall::open:return): invalid user access in action #5 at DIF offset 0
    85459/0xc0d24a:  fork()		 = 0 0
    dtrace: error on enabled probe ID 2170 (ID 163: syscall::open:return): invalid user access in action #5 at DIF offset 0




* https://apple.stackexchange.com/questions/343423/opensnoop-dtrace-error-on-enabled-probe-id-5-id-163-syscallopenreturn-i
* https://stackoverflow.com/questions/33476432/is-there-a-workaround-for-dtrace-cannot-control-executables-signed-with-restri

::

    csrutil enable --without dtrace    # from recovery mode (cmd R at startup)






DYLD_PRINT_LIBRARIES
-----------------------

::

    epsilon:opticks blyth$ DYLD_PRINT_LIBRARIES=1 OKTest 
    dyld: loaded: /usr/local/opticks/lib/OKTest
    dyld: loaded: /usr/local/opticks/lib/libOK.dylib
    dyld: loaded: /usr/local/opticks/lib/libOpticksGL.dylib
    dyld: loaded: /usr/local/opticks/lib/libOGLRap.dylib
    dyld: loaded: /usr/local/opticks/externals/lib/libGLEW.1.13.0.dylib
    dyld: loaded: /usr/local/opticks/externals/lib/libImGui.dylib
    dyld: loaded: /usr/local/opticks/externals/lib/libglfw.3.dylib
    dyld: loaded: /System/Library/Frameworks/Cocoa.framework/Versions/A/Cocoa
    dyld: loaded: /System/Library/Frameworks/OpenGL.framework/Versions/A/OpenGL
    dyld: loaded: /System/Library/Frameworks/IOKit.framework/Versions/A/IOKit
    dyld: loaded: /System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation
    dyld: loaded: /System/Library/Frameworks/CoreVideo.framework/Versions/A/CoreVideo
    dyld: loaded: /usr/local/opticks/lib/libOKOP.dylib
    dyld: loaded: /usr/local/opticks/lib/libOptiXRap.dylib
    dyld: loaded: /usr/local/opticks/lib/libOpticksGeo.dylib
    dyld: loaded: /usr/local/opticks/lib/libOpenMeshRap.dylib
    dyld: loaded: /usr/local/opticks/lib/libGGeo.dylib
    dyld: loaded: /usr/local/opticks/lib/libYoctoGLRap.dylib
    dyld: loaded: /usr/local/optix/lib64/liboptix.1.dylib
    dyld: loaded: /usr/local/optix/lib64/liboptixu.1.dylib
    dyld: loaded: /usr/local/optix/lib64/liboptix_prime.1.dylib
    dyld: loaded: /usr/local/opticks/lib/libThrustRap.dylib
    dyld: loaded: /usr/local/opticks/lib/libOpticksCore.dylib
    dyld: loaded: /usr/local/opticks/lib/libNPY.dylib
    dyld: loaded: /usr/local/opticks/lib/libBoostRap.dylib
    dyld: loaded: /usr/local/opticks_externals/boost/lib/libboost_system.dylib
    dyld: loaded: /usr/local/opticks_externals/boost/lib/libboost_program_options.dylib
    dyld: loaded: /usr/local/opticks_externals/boost/lib/libboost_filesystem.dylib
    dyld: loaded: /usr/local/opticks_externals/boost/lib/libboost_regex.dylib
    dyld: loaded: /usr/local/opticks/externals/lib/libOpenMeshCore.7.1.dylib
    dyld: loaded: /usr/local/opticks/externals/lib/libOpenMeshTools.7.1.dylib
    dyld: loaded: /usr/local/opticks/externals/lib/libYoctoGL.dylib
    dyld: loaded: /usr/lib/libc++.1.dylib
    dyld: loaded: /usr/local/opticks/externals/lib/libImplicitMesher.dylib
    dyld: loaded: /usr/local/opticks/externals/lib/libDualContouringSample.dylib
    dyld: loaded: /usr/local/opticks/lib/libCUDARap.dylib
    dyld: loaded: /usr/local/opticks/lib/libSysRap.dylib
    dyld: loaded: /usr/local/opticks/lib/libOKConf.dylib
    dyld: loaded: /usr/local/cuda/lib/libcurand.9.1.dylib
    dyld: loaded: /usr/lib/libSystem.B.dylib
    dyld: loaded: /System/Library/Frameworks/ApplicationServices.framework/Versions/A/ApplicationServices
    dyld: loaded: /System/Library/Frameworks/Security.framework/Versions/A/Security
    dyld: loaded: /System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGLU.dylib
    dyld: loaded: /System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGFXShared.dylib
    dyld: loaded: /usr/lib/libbsm.0.dylib
    dyld: loaded: /System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib

    ... several hundred system libs elided ...

    dyld: loaded: /System/Library/PrivateFrameworks/Symbolication.framework/Versions/A/Symbolication
    dyld: loaded: /System/Library/PrivateFrameworks/AppleFSCompression.framework/Versions/A/AppleFSCompression
    dyld: loaded: /System/Library/PrivateFrameworks/SpeechRecognitionCore.framework/Versions/A/SpeechRecognitionCore
    dyld: loaded: /System/Library/Frameworks/AGL.framework/Versions/A/AGL
    dyld: loaded: /usr/lib/libncurses.5.4.dylib
    dyld: loaded: /System/Library/Frameworks/Carbon.framework/Versions/A/Carbon
    dyld: loaded: /System/Library/Frameworks/Carbon.framework/Versions/A/Frameworks/CommonPanels.framework/Versions/A/CommonPanels
    dyld: loaded: /System/Library/Frameworks/Carbon.framework/Versions/A/Frameworks/Help.framework/Versions/A/Help
    dyld: loaded: /System/Library/Frameworks/Carbon.framework/Versions/A/Frameworks/ImageCapture.framework/Versions/A/ImageCapture
    dyld: loaded: /System/Library/Frameworks/Carbon.framework/Versions/A/Frameworks/OpenScripting.framework/Versions/A/OpenScripting
    dyld: loaded: /System/Library/Frameworks/Carbon.framework/Versions/A/Frameworks/Print.framework/Versions/A/Print
    dyld: loaded: /System/Library/Frameworks/Carbon.framework/Versions/A/Frameworks/SecurityHI.framework/Versions/A/SecurityHI
    2020-11-24 10:30:50.925 INFO  [12606008] [BOpticksKey::SetKey@77]  spec OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3
    2020-11-24 10:30:50.926 INFO  [12606008] [Opticks::init@428] INTEROP_MODE hostname epsilon.local
    2020-11-24 10:30:50.927 INFO  [12606008] [Opticks::init@437]  mandatory keyed access to geometry, opticksaux 
    2020-11-24 10:30:50.930 INFO  [12606008] [BOpticksResource::setupViaKey@881] 
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3
                     exename  : OKX4Test
             current_exename  : OKTest
                       class  : X4PhysicalVolume
                     volname  : World0xc15cfc00x40f7000_PV
                      digest  : 50a18baaf29b18fae8c1642927003ee3
                      idname  : OKX4Test_World0xc15cfc00x40f7000_PV_g4live
                      idfile  : g4ok.gltf

