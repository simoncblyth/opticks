NNodeNudger-not-implemented-noise
======================================


Issue
--------

Getting lots of NNodeNudger noise in logs regarding minmin unimplemented with DYBXTMP geom

* the complaints are about the base geometry, and with test running that is not used : so this is just an annoyance here



::

    ts interlocked --oktest --pfx scan-px-10 --cat cvd_1_rtx_0_1M --generateoverride 1000000 --compute --production --savehit --multievent 10 --xanalytic --rngmax 3 --cvd 1 --rtx 0 -D


::

    [blyth@localhost npy]$ eo
    OPTICKS_INSTALL_PREFIX=/home/blyth/local/opticks
    OPTICKS_EVENT_BASE=/home/blyth/local/opticks/evtbase

    OPTICKS_KEY_DYBXTMP=OKX4Test.X4PhysicalVolume.World0xc15cfc00x4552410_PV.5aa828335373870398bf4f738781da6c
    OPTICKS_KEY_JV5=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
    OPTICKS_KEY=OKX4Test.X4PhysicalVolume.World0xc15cfc00x4552410_PV.5aa828335373870398bf4f738781da6c

    OPTICKS_COMPUTE_CAPABILITY=70
    OPTICKS_NVIDIA_DRIVER_VERSION=435.21
    OPTICKS_HOME=/home/blyth/opticks
    OPTICKS_LEGACY_GEOMETRY_ENABLED=1
    OPTICKS_ANA_DEFAULTS=det=g4live,cat=cvd_1_rtx_1_1M,src=torch,tag=1,pfx=scan-ph
    OPTICKS_DEFAULT_INTEROP_CVD=1
    OPTICKS_RESULTS_PREFIX=/home/blyth/local/opticks


* CAUTION this is running on DYBXTMP, need to adopt an OPTICKS_KEY with better audit that than one
  as the default key for tests 




Planting sigint::

    2019-09-23 11:06:24.813 INFO  [87145] [OpticksHub::loadGeometry@542] [ /home/blyth/.opticks/geocache/OKX4Test_World0xc15cfc00x4552410_PV_g4live/g4ok_gltf/5aa828335373870398bf4f738781da6c/1
    NNodeNudger::znudge_difference_minmin coin ( 3, 2) PAIR_MINMIN NUDGE_NONE [ 3:cy] P [ 2:co] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:06:25.322 FATAL [87145] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 

    Program received signal SIGINT, Interrupt.
    0x00007ffff0aa149b in raise () from /lib64/libpthread.so.0
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff0aa149b in raise () from /lib64/libpthread.so.0
    #1  0x00007ffff29d506c in NNodeNudger::znudge_difference_minmin (this=0x1f5b2c0, coin=0x1f598c0) at /home/blyth/opticks/npy/NNodeNudger.cpp:314
    #2  0x00007ffff29d4e24 in NNodeNudger::znudge (this=0x1f5b2c0, coin=0x1f598c0) at /home/blyth/opticks/npy/NNodeNudger.cpp:277
    #3  0x00007ffff29d4d74 in NNodeNudger::uncoincide (this=0x1f5b2c0) at /home/blyth/opticks/npy/NNodeNudger.cpp:267
    #4  0x00007ffff29d3de4 in NNodeNudger::init (this=0x1f5b2c0) at /home/blyth/opticks/npy/NNodeNudger.cpp:79
    #5  0x00007ffff29d3b49 in NNodeNudger::NNodeNudger (this=0x1f5b2c0, root_=0x1f5a450, epsilon_=9.99999975e-06) at /home/blyth/opticks/npy/NNodeNudger.cpp:60
    #6  0x00007ffff2a391af in NCSG::make_nudger (this=0x1f37140, msg=0x7ffff2b2f7fb "postimport") at /home/blyth/opticks/npy/NCSG.cpp:1372
    #7  0x00007ffff2a3551f in NCSG::postimport (this=0x1f37140) at /home/blyth/opticks/npy/NCSG.cpp:429
    #8  0x00007ffff2a353ad in NCSG::import (this=0x1f37140) at /home/blyth/opticks/npy/NCSG.cpp:420
    #9  0x00007ffff2a33cf1 in NCSG::Load (treedir=0x1e3ffb8 "/home/blyth/.opticks/geocache/OKX4Test_World0xc15cfc00x4552410_PV_g4live/g4ok_gltf/5aa828335373870398bf4f738781da6c/1/GMeshLibNCSG/29", config=0x1f56ef0)
            at /home/blyth/opticks/npy/NCSG.cpp:104
    #10 0x00007ffff2a339df in NCSG::Load (treedir=0x1e3ffb8 "/home/blyth/.opticks/geocache/OKX4Test_World0xc15cfc00x4552410_PV_g4live/g4ok_gltf/5aa828335373870398bf4f738781da6c/1/GMeshLibNCSG/29")
            at /home/blyth/opticks/npy/NCSG.cpp:73
    #11 0x00007ffff50e708f in GMeshLib::loadMeshes (this=0x1a59dd0, idpath=0x68b5a0 "/home/blyth/.opticks/geocache/OKX4Test_World0xc15cfc00x4552410_PV_g4live/g4ok_gltf/5aa828335373870398bf4f738781da6c/1")
            at /home/blyth/opticks/ggeo/GMeshLib.cc:473
    #12 0x00007ffff50e560c in GMeshLib::loadFromCache (this=0x1a59dd0) at /home/blyth/opticks/ggeo/GMeshLib.cc:75
    #13 0x00007ffff50e5561 in GMeshLib::Load (ok=0x65c470) at /home/blyth/opticks/ggeo/GMeshLib.cc:64
    #14 0x00007ffff50da82c in GGeo::loadFromCache (this=0x68e2e0) at /home/blyth/opticks/ggeo/GGeo.cc:898
    #15 0x00007ffff50d8b63 in GGeo::loadGeometry (this=0x68e2e0) at /home/blyth/opticks/ggeo/GGeo.cc:626
    #16 0x00007ffff64e1dcf in OpticksGeometry::loadGeometryBase (this=0x690050) at /home/blyth/opticks/opticksgeo/OpticksGeometry.cc:156
    #17 0x00007ffff64e17f3 in OpticksGeometry::loadGeometry (this=0x690050) at /home/blyth/opticks/opticksgeo/OpticksGeometry.cc:98
    #18 0x00007ffff64e64da in OpticksHub::loadGeometry (this=0x678120) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:546
    #19 0x00007ffff64e4f1e in OpticksHub::init (this=0x678120) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:253
    #20 0x00007ffff64e4c0f in OpticksHub::OpticksHub (this=0x678120, ok=0x65c470) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:217
    #21 0x00007ffff7bd59cf in OKMgr::OKMgr (this=0x7fffffffcb90, argc=44, argv=0x7fffffffcd08, argforced=0x0) at /home/blyth/opticks/ok/OKMgr.cc:54
    #22 0x0000000000402ead in main (argc=44, argv=0x7fffffffcd08) at /home/blyth/opticks/ok/tests/OKTest.cc:32
    (gdb) 





::

    tboolean-;NCSG=ERROR tboolean-interlocked --oktest


A few solids with lots of minmin coincidence::

    2019-09-23 11:14:47.494 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 27 soname CtrGdsOflTfbInLso0xbfa2d300x42d90d0 treeNameIdx 27
    2019-09-23 11:14:47.497 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 28 soname CtrGdsOflInLso0xbfa11780x42d91f0 treeNameIdx 28
    2019-09-23 11:14:47.499 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 29 soname OcrGdsPrt0xc3525180x42d9ec0 treeNameIdx 29
    NNodeNudger::znudge_difference_minmin coin ( 3, 2) PAIR_MINMIN NUDGE_NONE [ 3:cy] P [ 2:co] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:47.499 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    2019-09-23 11:14:47.501 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 30 soname OcrGdsTfbInLso0xbfa23700x42da2d0 treeNameIdx 30
    NNodeNudger::znudge_difference_minmin coin ( 1, 5) PAIR_MINMIN NUDGE_NONE [ 1:co] P [ 5:cy] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:47.501 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    2019-09-23 11:14:47.525 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 31 soname OcrGdsInLso0xbfa21900x42daab0 treeNameIdx 31
    NNodeNudger::znudge_difference_minmin coin ( 3, 2) PAIR_MINMIN NUDGE_NONE [ 3:co] P [ 2:cy] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:47.525 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    2019-09-23 11:14:47.550 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 32 soname OavBotRib0xbfaafe00x42cd130 treeNameIdx 32
    2019-09-23 11:14:47.551 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 33 soname OavBotHub0xc3550300x42cd1f0 treeNameIdx 33
    2019-09-23 11:14:47.552 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 34 soname IavBotRib0xc2cd8b80x42cd190 treeNameIdx 34
    2019-09-23 11:14:47.554 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 35 soname IavBotHub0xbf8cfd00x42db1e0 treeNameIdx 35


    2019-09-23 11:14:49.858 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 80 soname CtrLsoOflInOil0xc1831a00x42f11c0 treeNameIdx 80
    2019-09-23 11:14:49.859 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 81 soname OcrGdsLsoPrt0xc1049780x42f1e20 treeNameIdx 81
    NNodeNudger::znudge_difference_minmin coin ( 3, 2) PAIR_MINMIN NUDGE_NONE [ 3:cy] P [ 2:co] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:49.859 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    2019-09-23 11:14:49.862 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 82 soname OcrGdsInLsoOfl0xc26f4500x42f2270 treeNameIdx 82
    2019-09-23 11:14:49.863 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 83 soname OcrGdsTfbInLsoOfl0xc2b5ba00x42f26f0 treeNameIdx 83
    2019-09-23 11:14:49.865 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 84 soname OcrGdsLsoInOil0xc5407380x42f2b70 treeNameIdx 84
    2019-09-23 11:14:49.867 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 85 soname OcrCalLsoPrt0xc1076b00x42f3980 treeNameIdx 85
    NNodeNudger::znudge_difference_minmin coin ( 3, 2) PAIR_MINMIN NUDGE_NONE [ 3:cy] P [ 2:co] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:49.867 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    2019-09-23 11:14:49.869 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 86 soname OcrCalLso0xc103c180x42f3d90 treeNameIdx 86
    2019-09-23 11:14:49.871 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 87 soname WallLedDiffuserBall0xc3aa0d00x42f3f50 treeNameIdx 87
    2019-09-23 11:14:49.872 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 88 soname wall-led-rod0xc3479700x42f40a0 treeNameIdx 88
    2019-09-23 11:14:49.873 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 89 soname wall-led-assy0xc3a99a00x42e34f0 treeNameIdx 89
    2019-09-23 11:14:49.875 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 90 soname oil0xbf5ed480x42e3710 treeNameIdx 90
    2019-09-23 11:14:49.876 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 91 soname CenterCalibHoleSST0xbf766100x42e37f0 treeNameIdx 91
    2019-09-23 11:14:49.877 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 92 soname OffCenterCalibHoleSST0xc21d2d00x42e38d0 treeNameIdx 92
    2019-09-23 11:14:49.878 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 93 soname GCatCalibHoleSST0xc345f880x42e39b0 treeNameIdx 93
    2019-09-23 11:14:49.879 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 94 soname sst0xbf4b0600x42e3ac0 treeNameIdx 94
    2019-09-23 11:14:49.880 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 95 soname BottomPlate0xc3a40600x42e3be0 treeNameIdx 95
    2019-09-23 11:14:49.881 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 96 soname ShieldingPuck0xc0ad1780x42e3d00 treeNameIdx 96
    2019-09-23 11:14:49.882 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 97 soname BearingRing0xbf778c80x42f5ad0 treeNameIdx 97
    2019-09-23 11:14:49.886 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 98 soname turntable0xbf784f00x42f65f0 treeNameIdx 98
    NNodeNudger::znudge_difference_minmin coin ( 7, 8) PAIR_MINMIN NUDGE_NONE [ 7:cy] P [ 8:cy] P sibs Y u_sibs N u_par N u_same N 
    2019-09-23 11:14:49.886 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin ( 7, 4) PAIR_MINMIN NUDGE_NONE [ 7:cy] P [ 4:cy] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:49.886 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin ( 7, 2) PAIR_MINMIN NUDGE_NONE [ 7:cy] P [ 2:cy] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:49.886 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin ( 8, 4) PAIR_MINMIN NUDGE_NONE [ 8:cy] P [ 4:cy] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:49.886 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin ( 8, 2) PAIR_MINMIN NUDGE_NONE [ 8:cy] P [ 2:cy] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:49.886 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin ( 4, 2) PAIR_MINMIN NUDGE_NONE [ 4:cy] P [ 2:cy] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:49.886 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    2019-09-23 11:14:49.887 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 99 soname DiffuserBall0xc3073d00x42f67b0 treeNameIdx 99
    2019-09-23 11:14:49.889 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 100 soname led-source-shell0xc3068f00x42f7080 treeNameIdx 100
    2019-09-23 11:14:49.891 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 101 soname Weight0xc3083a00x42f72a0 treeNameIdx 101


    2019-09-23 11:14:50.313 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 56 soname RadialShieldUnit0xc3d7da80x42db120 treeNameIdx 250
    2019-09-23 11:14:50.480 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 57 soname TopESRCutHols0xbf9de100x42e5980 treeNameIdx 251
    NNodeNudger::znudge_difference_minmin coin (256,128) PAIR_MINMIN NUDGE_NONE [256:di] P [128:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.480 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (256,64) PAIR_MINMIN NUDGE_NONE [256:di] P [64:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.480 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (256,32) PAIR_MINMIN NUDGE_NONE [256:di] P [32:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.480 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (256,16) PAIR_MINMIN NUDGE_NONE [256:di] P [16:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.480 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (256, 8) PAIR_MINMIN NUDGE_NONE [256:di] P [ 8:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.480 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (256, 4) PAIR_MINMIN NUDGE_NONE [256:di] P [ 4:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.480 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (256, 2) PAIR_MINMIN NUDGE_NONE [256:di] P [ 2:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.480 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (128,64) PAIR_MINMIN NUDGE_NONE [128:di] P [64:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.480 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (128,32) PAIR_MINMIN NUDGE_NONE [128:di] P [32:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.480 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (128,16) PAIR_MINMIN NUDGE_NONE [128:di] P [16:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.480 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (128, 8) PAIR_MINMIN NUDGE_NONE [128:di] P [ 8:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.480 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (128, 4) PAIR_MINMIN NUDGE_NONE [128:di] P [ 4:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.480 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (128, 2) PAIR_MINMIN NUDGE_NONE [128:di] P [ 2:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.480 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (64,32) PAIR_MINMIN NUDGE_NONE [64:di] P [32:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.481 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (64,16) PAIR_MINMIN NUDGE_NONE [64:di] P [16:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.481 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (64, 8) PAIR_MINMIN NUDGE_NONE [64:di] P [ 8:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.481 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (64, 4) PAIR_MINMIN NUDGE_NONE [64:di] P [ 4:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.481 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (64, 2) PAIR_MINMIN NUDGE_NONE [64:di] P [ 2:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.481 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (32,16) PAIR_MINMIN NUDGE_NONE [32:di] P [16:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.481 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (32, 8) PAIR_MINMIN NUDGE_NONE [32:di] P [ 8:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.481 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (32, 4) PAIR_MINMIN NUDGE_NONE [32:di] P [ 4:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.481 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (32, 2) PAIR_MINMIN NUDGE_NONE [32:di] P [ 2:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.481 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (16, 8) PAIR_MINMIN NUDGE_NONE [16:di] P [ 8:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.481 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (16, 4) PAIR_MINMIN NUDGE_NONE [16:di] P [ 4:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.481 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (16, 2) PAIR_MINMIN NUDGE_NONE [16:di] P [ 2:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.481 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin ( 8, 4) PAIR_MINMIN NUDGE_NONE [ 8:di] P [ 4:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.481 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin ( 8, 2) PAIR_MINMIN NUDGE_NONE [ 8:di] P [ 2:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.481 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin ( 4, 2) PAIR_MINMIN NUDGE_NONE [ 4:di] P [ 2:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.481 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    2019-09-23 11:14:50.493 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 58 soname TopRefGapCutHols0xbf9cef80x42e6860 treeNameIdx 252
    NNodeNudger::znudge_difference_minmin coin (16, 8) PAIR_MINMIN NUDGE_NONE [16:di] P [ 8:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.493 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (16, 4) PAIR_MINMIN NUDGE_NONE [16:di] P [ 4:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.493 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (16, 2) PAIR_MINMIN NUDGE_NONE [16:di] P [ 2:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.493 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin ( 8, 4) PAIR_MINMIN NUDGE_NONE [ 8:di] P [ 4:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.493 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin ( 8, 2) PAIR_MINMIN NUDGE_NONE [ 8:di] P [ 2:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.493 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin ( 4, 2) PAIR_MINMIN NUDGE_NONE [ 4:di] P [ 2:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.493 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    2019-09-23 11:14:50.498 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 59 soname TopRefCutHols0xbf9bd500x42e7710 treeNameIdx 253
    NNodeNudger::znudge_difference_minmin coin (16, 8) PAIR_MINMIN NUDGE_NONE [16:cy] P [ 8:cy] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.498 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (16, 4) PAIR_MINMIN NUDGE_NONE [16:cy] P [ 4:cy] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.498 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin (16, 2) PAIR_MINMIN NUDGE_NONE [16:cy] P [ 2:cy] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.498 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin ( 8, 4) PAIR_MINMIN NUDGE_NONE [ 8:cy] P [ 4:cy] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.498 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin ( 8, 2) PAIR_MINMIN NUDGE_NONE [ 8:cy] P [ 2:cy] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.498 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin ( 4, 2) PAIR_MINMIN NUDGE_NONE [ 4:cy] P [ 2:cy] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.498 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    2019-09-23 11:14:50.505 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 60 soname BotESRCutHols0xbfa73680x42e8e30 treeNameIdx 254
    NNodeNudger::znudge_difference_minmin coin ( 8, 4) PAIR_MINMIN NUDGE_NONE [ 8:di] P [ 4:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.505 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin ( 8, 2) PAIR_MINMIN NUDGE_NONE [ 8:di] P [ 2:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.505 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    NNodeNudger::znudge_difference_minmin coin ( 4, 2) PAIR_MINMIN NUDGE_NONE [ 4:di] P [ 2:di] P sibs N u_sibs N u_par N u_same N 
    2019-09-23 11:14:50.505 FATAL [89972] [NNodeNudger::znudge_difference_minmin@312]  NOT IMPLEMENTED 
    2019-09-23 11:14:50.521 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 61 soname BotRefGapCutHols0xc34bb280x42e9b20 treeNameIdx 255
    2019-09-23 11:14:50.530 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 62 soname BotRefHols0xc3cd3800x42ea7c0 treeNameIdx 256
    2019-09-23 11:14:50.533 ERROR [89972] [NCSG::make_nudger@1370]  lvIdx 65 soname SstBotCirRibBase0xc26e2d00x42eba20 treeNameIdx 257



