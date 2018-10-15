4/120 NPY linux test fails, no fails on macOS
======================================================


issue : NOW FIXED
---------------------


::

    totals  4   / 372 


    FAILS:
      86 /120 Test #86 : NPYTest.NCSGLoadTest                          ***Exception: Child aborted    0.06   
      93 /120 Test #93 : NPYTest.NScanTest                             ***Exception: Child aborted    0.07   
      113/120 Test #113: NPYTest.NSceneTest                            ***Exception: Child aborted    1.72   
      117/120 Test #117: NPYTest.NSceneMeshTest                        ***Exception: Child aborted    1.55   



Summary
-----------

Cause:

* existing persisted buffers name srcnodes.npy not matching code, which was a side-effect of the 
  migration to direct geometry  

Solution:

* recreate extras and geocache with gdml2gltf

Bonus:

* revives analytic DYB raytrace, in legacy workflow



All 4 from same cause, lask of non-optionsl srcnodes.npy 
-----------------------------------------------------------

Looks like the geocache extras is not uptodate with the code.

::

    blyth@localhost extras]$ pwd
    /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras


::

    [blyth@localhost issues]$ NCSGLoadTest
    2018-10-15 13:38:31.262 ERROR [22199] [BOpticksResource::init@87] layout : 0
    2018-10-15 13:38:31.263 INFO  [22199] [main@92] basedir:/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras
    2018-10-15 13:38:31.263 WARN  [22199] [BStr::atoi@264] bad_lexical_cast [bad lexical cast: source type value could not be interpreted as target]  with [extras]
    2018-10-15 13:38:31.263 ERROR [22199] [BStr::atoi@276] BStr::atoi badlex  str extras fallback -666
    2018-10-15 13:38:31.264 INFO  [22199] [NCSGList::load@133] NCSGList::load VERBOSITY 0 basedir /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras txtpath /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/csg.txt nbnd 249
    2018-10-15 13:38:31.264 INFO  [22199] [NSceneConfig::env_override@82] NSceneConfig override verbosity from VERBOSITY envvar 1
    2018-10-15 13:38:31.264 FATAL [22199] [NPYList::loadBuffer@88]  non-optional buffer does not exist /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/248/srcnodes.npy
    NCSGLoadTest: /home/blyth/opticks/npy/NPYList.cpp:89: void NPYList::loadBuffer(const char*, int, const char*): Assertion `0' failed.
    Aborted (core dumped)

::

    Program received signal SIGABRT, Aborted.
    0x00007ffff48bf277 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 glibc-2.17-222.el7.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-19.el7.x86_64 libcom_err-1.42.9-12.el7_5.x86_64 libgcc-4.8.5-28.el7_5.1.x86_64 libicu-50.1.2-15.el7.x86_64 libselinux-2.5-12.el7.x86_64 libstdc++-4.8.5-28.el7_5.1.x86_64 openssl-libs-1.0.2k-12.el7.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-17.el7.x86_64
    (gdb) bt
    0  0x00007ffff48bf277 in raise () from /lib64/libc.so.6
    1  0x00007ffff48c0968 in abort () from /lib64/libc.so.6
    2  0x00007ffff48b8096 in __assert_fail_base () from /lib64/libc.so.6
    3  0x00007ffff48b8142 in __assert_fail () from /lib64/libc.so.6
    4  0x00007ffff79189e0 in NPYList::loadBuffer (this=0x6166f0, treedir=0x616a80 "/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/248", bid=0, msg=0x0) at /home/blyth/opticks/npy/NPYList.cpp:89
    5  0x00007ffff79e5e24 in NCSGData::loadsrc (this=0x6163f0, treedir=0x616a80 "/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/248") at /home/blyth/opticks/npy/NCSGData.cpp:81
    6  0x00007ffff79dee4d in NCSG::loadsrc (this=0x616640) at /home/blyth/opticks/npy/NCSG.cpp:221
    7  0x00007ffff79de429 in NCSG::Load (treedir=0x615dc8 "/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/248", config=0x615e70) at /home/blyth/opticks/npy/NCSG.cpp:77
    8  0x00007ffff79de123 in NCSG::Load (treedir=0x615dc8 "/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/248") at /home/blyth/opticks/npy/NCSG.cpp:47
    9  0x00007ffff79ea1a0 in NCSGList::loadTree (this=0x615800, idx=248, boundary=0x61b2f8 "extras/245") at /home/blyth/opticks/npy/NCSGList.cpp:254
    10 0x00007ffff79e9b67 in NCSGList::load (this=0x615800) at /home/blyth/opticks/npy/NCSGList.cpp:156
    11 0x00007ffff79e92f7 in NCSGList::Load (csgpath=0x614f10 "/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras", verbosity=0, checkmaterial=false) at /home/blyth/opticks/npy/NCSGList.cpp:40
    12 0x0000000000404fbb in main (argc=1, argv=0x7fffffffda38) at /home/blyth/opticks/npy/tests/NCSGLoadTest.cc:116
    (gdb) 



::

    [blyth@localhost extras]$ ll 248/
    total 32
    drwxrwxr-x.   2 blyth blyth   27 Aug 30 15:37 0
    -rw-rw-r--.   1 blyth blyth 1149 Aug 30 15:37 NNodeTest_248.cc
    -rw-rw-r--.   1 blyth blyth  124 Aug 30 15:37 meta.json
    -rw-rw-r--.   1 blyth blyth 2743 Aug 30 15:37 tbool248.bash
    -rw-rw-r--.   1 blyth blyth  192 Aug 30 15:37 nodes.npy
    -rw-rw-r--.   1 blyth blyth  192 Aug 30 15:37 transforms.npy
    drwxrwxr-x.   3 blyth blyth  116 Aug 30 15:37 .
    drwxrwxr-x. 251 blyth blyth 8192 Aug 30 15:37 ..
    [blyth@localhost extras]$ 



Try to update geocache
--------------------------

::

    OKTest --gltf 3 -G 

But it runs into the same problem at::

    2018-10-15 13:49:38.656 FATAL [22585] [NPYList::loadBuffer@88]  non-optional buffer does not exist /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/248/srcnodes.npy
    OKTest: /home/blyth/opticks/npy/NPYList.cpp:89: void NPYList::loadBuffer(const char*, int, const char*): Assertion `0' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffe7f75277 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-222.el7.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-19.el7.x86_64 libX11-1.6.5-1.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.14-8.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-12.el7_5.x86_64 libgcc-4.8.5-28.el7_5.1.x86_64 libicu-50.1.2-15.el7.x86_64 libselinux-2.5-12.el7.x86_64 libstdc++-4.8.5-28.el7_5.1.x86_64 libxcb-1.12-1.el7.x86_64 openssl-libs-1.0.2k-12.el7.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-17.el7.x86_64
    (gdb) bt
    0  0x00007fffe7f75277 in raise () from /lib64/libc.so.6
    1  0x00007fffe7f76968 in abort () from /lib64/libc.so.6
    2  0x00007fffe7f6e096 in __assert_fail_base () from /lib64/libc.so.6
    3  0x00007fffe7f6e142 in __assert_fail () from /lib64/libc.so.6
    4  0x00007fffef8179e0 in NPYList::loadBuffer (this=0xb064620, treedir=0xb061740 "/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/248", bid=0, msg=0x0)
        at /home/blyth/opticks/npy/NPYList.cpp:89
    5  0x00007fffef8e4e24 in NCSGData::loadsrc (this=0xb0614e0, treedir=0xb061740 "/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/248") at /home/blyth/opticks/npy/NCSGData.cpp:81
    6  0x00007fffef8dde4d in NCSG::loadsrc (this=0xb064570) at /home/blyth/opticks/npy/NCSG.cpp:221
    7  0x00007fffef8dd429 in NCSG::Load (treedir=0xb0643d8 "/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/248", config=0x69a3b0) at /home/blyth/opticks/npy/NCSG.cpp:77
    8  0x00007fffef96ce39 in NGLTF::getCSG (this=0x9222310, mesh_id=0) at /home/blyth/opticks/npy/NGLTF.cpp:368
    9  0x00007fffef97e011 in NScene::load_mesh_extras (this=0x957c820) at /home/blyth/opticks/npy/NScene.cpp:488
    10 0x00007fffef97c2a3 in NScene::init (this=0x957c820) at /home/blyth/opticks/npy/NScene.cpp:218
    11 0x00007fffef97bf22 in NScene::NScene (this=0x957c820, source=0x9222310, idfold=0x680130 "/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300", dbgnode=-1) at /home/blyth/opticks/npy/NScene.cpp:167
    12 0x00007fffef97bb64 in NScene::Load (base=0x92310a0 "/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300", name=0x91e8e80 "g4_00.gltf", 
        idfold=0x680130 "/home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300", config=0x69a3b0, dbgnode=-1, scene_idx=0) at /home/blyth/opticks/npy/NScene.cpp:118
    13 0x00007ffff50fbe38 in GScene::GScene (this=0x9334c10, ok=0x6643c0, ggeo=0x68a880, loaded=false) at /home/blyth/opticks/ggeo/GScene.cc:127
    14 0x00007ffff50fbaeb in GScene::Create (ok=0x6643c0, ggeo=0x68a880) at /home/blyth/opticks/ggeo/GScene.cc:74
    15 0x00007ffff50ed106 in GGeo::loadAnalyticFromGLTF (this=0x68a880) at /home/blyth/opticks/ggeo/GGeo.cc:674
    16 0x00007ffff50ec8be in GGeo::loadGeometry (this=0x68a880) at /home/blyth/opticks/ggeo/GGeo.cc:574
    17 0x00007ffff64ead79 in OpticksGeometry::loadGeometryBase (this=0x688780) at /home/blyth/opticks/opticksgeo/OpticksGeometry.cc:139
    18 0x00007ffff64ea7a1 in OpticksGeometry::loadGeometry (this=0x688780) at /home/blyth/opticks/opticksgeo/OpticksGeometry.cc:89
    19 0x00007ffff64ef1f2 in OpticksHub::loadGeometry (this=0x680dd0) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:407
    20 0x00007ffff64edd7a in OpticksHub::init (this=0x680dd0) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:177
    21 0x00007ffff64edb9a in OpticksHub::OpticksHub (this=0x680dd0, ok=0x6643c0) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:156
    22 0x00007ffff7bd585f in OKMgr::OKMgr (this=0x7fffffffd870, argc=4, argv=0x7fffffffd9e8, argforced=0x0) at /home/blyth/opticks/ok/OKMgr.cc:44
    23 0x0000000000402e8b in main (argc=4, argv=0x7fffffffd9e8) at /home/blyth/opticks/ok/tests/OKTest.cc:13
    (gdb) 


Try moving aside the entire geocache::

   blyth@localhost DayaBay_VGDX_20140414-1300]$ mv g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae_0

This makes no difference, the extras are not within the digest directory. They get written by the  python machinery... hmm need to find the old
ab testing, found it in ab- but that doesnt fo back to the gdml2gltf running. 


Recreate extras with gdml2gltf
------------------------------------

Try moving aside extras too::

    [blyth@localhost DayaBay_VGDX_20140414-1300]$ l
    total 17696
    drwxrwxr-x.  12 blyth blyth     209 Oct 15 13:56 g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    drwxrwxr-x.   2 blyth blyth       6 Oct 15 13:47 g4_00
    drwxrwxr-x. 251 blyth blyth    8192 Oct 15 13:47 extras
    drwxrwxr-x.  13 blyth blyth     234 Oct 15 13:20 g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae_0
    -rw-rw-r--.   1 blyth blyth 4200687 Aug 30 15:37 g4_00.gltf
    -rw-rw-r--.   1 blyth blyth 2663880 Jul  5 17:31 g4_00.idmap
    -rw-rw-r--.   1 blyth blyth 4111332 Jul  5 17:31 g4_00.gdml
    -rw-rw-r--.   1 blyth blyth 7126305 Jul  5 17:31 g4_00.dae
    [blyth@localhost DayaBay_VGDX_20140414-1300]$ mv extras extras_0

And recreate it with::

    op --gdml2gltf

Now recreate geocache::

    OKTest --gltf 3 -G

Now can see analytic raytrace with::
   
   OKTest --gltf 3 --tracer        

   OKTest --tracer      ## without the gltf option get the triangulated geometry 




NCSGLoadTest has rc 0 but much output
----------------------------------------

::


    2018-10-15 14:24:04.501 ERROR [24120] [NPYList::setBuffer@122] replacing nodes.npy buffer  prior 1,4,4 buffer 1,4,4 msg prepareForExport
    2018-10-15 14:24:04.501 ERROR [24120] [NPYList::setBuffer@122] replacing planes.npy buffer  prior 0,4 buffer 0,4 msg prepareForExport
    2018-10-15 14:24:04.501 ERROR [24120] [NPYList::setBuffer@122] replacing idx.npy buffer  prior 1,4 buffer 1,4 msg prepareForExport


npy-t confirms the issue is FIXED
---------------------------------------


