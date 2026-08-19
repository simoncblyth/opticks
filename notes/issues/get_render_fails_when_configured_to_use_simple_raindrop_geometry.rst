get_render_fails_when_configured_to_use_simple_raindrop_geometry
==================================================================

* simple fixes by changing default MOI to "", and coping with out TNUDGE



::

    FAILS:  2   / 223   :  Wed Aug 19 17:35:59 2026  :  GEOM RaindropRockAirWater
      102/110 Test #102: SysRapTest.SGLFW_SOPTIX_Scene_test                      ***Failed                      0.11
      3  /4   Test #3  : CSGOptiXTest.CSGOptiXRenderTest                         ***Failed                      0.79


::

    102/110 Test #102: SysRapTest.SGLFW_SOPTIX_Scene_test .......................***Failed    0.10 sec
                    HOME : /home/blyth
                     PWD : /data1/blyth/local/opticks_Debug/build/sysrap/tests
                    GEOM : RaindropRockAirWater
             BASH_SOURCE : /data1/blyth/local/opticks_Debug/bin/STestRunner.sh
              EXECUTABLE : SGLFW_SOPTIX_Scene_test
                    ARGS :
    /data1/blyth/local/opticks_Debug/bin/STestRunner.sh: line 68: 879514 Segmentation fault      (core dumped) $EXECUTABLE $@
    /data1/blyth/local/opticks_Debug/bin/STestRunner.sh : FAIL from SGLFW_SOPTIX_Scene_test



SGLFW_SOPTIX_Scene_test - is from lack of TNUDGE ekey
-----------------------------------------------------

This could be lack of TNUDGE ?::

    Starting program: /data1/blyth/local/opticks_Debug/lib/SGLFW_SOPTIX_Scene_test
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    [Detaching after vfork from child process 880299]
    NP_slice::parse {::100} -> start: 0, stop: 9223372036854775807, step: 100

    Program received signal SIGSEGV, Segmentation fault.
    0x00007ffff644097b in getenv () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64 libX11-1.7.0-11.el9.x86_64 libXau-1.0.9-8.el9.x86_64 libXext-1.3.4-8.el9.x86_64 libgcc-11.5.0-5.el9_5.alma.1.x86_64 libglvnd-1.3.4-1.el9.x86_64 libglvnd-glx-1.3.4-1.el9.x86_64 libstdc++-11.5.0-5.el9_5.alma.1.x86_64 libxcb-1.13.1-9.el9.x86_64 openssl-libs-3.5.1-7.el9_7.x86_64
    (gdb) bt
    #0  0x00007ffff644097b in getenv () from /lib64/libc.so.6
    #1  0x000000000044f5f0 in ssys::getenv_<float> (ekey=0x0, fallback=0) at /home/blyth/opticks/sysrap/tests/../ssys.h:668
    #2  0x000000000044f00c in ssys::getenvfloat (ekey=0x0, fallback=0) at /home/blyth/opticks/sysrap/tests/../ssys.h:521
    #3  0x0000000000494a60 in SRecord::Load (_fold=0x51d74c "$AFOLD", _slice=0x51d738 "$AFOLD_RECORD_SLICE", _dt=0x0) at /home/blyth/opticks/sysrap/tests/../SRecord.h:163
    #4  0x0000000000445fb7 in main (argc=1, argv=0x7fffffffb698) at /home/blyth/opticks/sysrap/tests/SGLFW_SOPTIX_Scene_test.cc:58
    (gdb)


YEP, FIXED




CSGOptiXRenderTest - better handling of frame spec that is not valid for the geometry ?
----------------------------------------------------------------------------------------

::

    SGLM::initView VIEW [-] load_interpolated_view NO  interpolated_view.brief -
    stree::parse_spec FAILED to find lvid for q_soname [sWorld]
    [stree::desc_soname
    [G4_WATER_solid]
    [G4_AIR_solid]
    [G4_Pb_solid]
    [VACUUM_solid]
    ]stree::desc_soname
    stree::get_frame FATAL parse_spec failed  q_spec [sWorld:0:0] parse_rc -1
    CSGOptiXRenderTest: /data1/blyth/local/opticks_Debug/include/SysRap/stree.h:2986: int stree::get_frame_from_triplet(sfr&, const char*) const: Assertion `parse_rc == 0' failed.
    /data1/blyth/local/opticks_Debug/bin/CXTestRunner.sh: line 48: 879671 Aborted                 (core dumped) $EXECUTABLE $@
    /data1/blyth/local/opticks_Debug/bin/CXTestRunner.sh : FAIL from CSGOptiXRenderTest

        Start 4: CSGOptiXTest.ParamsTest
    4/4 Test #4: CSGOptiXTest.ParamsTest ............   Passed    0.02 sec

    75% tests passed, 1 tests failed out of 4

    Total Test time (real) =   0.84 sec


::

    [lo] A[blyth@localhost tests]$ ~/o/CSGOptiX/tests/CSGOptiXRenderTest.sh dbg
    gdb -ex r --args CSGOptiXRenderTest
    ...
    SGLM::initView VIEW [-] load_interpolated_view NO  interpolated_view.brief -
    stree::parse_spec FAILED to find lvid for q_soname [sWorld]
    [stree::desc_soname
    [G4_WATER_solid]
    [G4_AIR_solid]
    [G4_Pb_solid]
    [VACUUM_solid]
    ]stree::desc_soname
    stree::get_frame FATAL parse_spec failed  q_spec [sWorld:0:0] parse_rc -1
    CSGOptiXRenderTest: /data1/blyth/local/opticks_Debug/include/SysRap/stree.h:2986: int stree::get_frame_from_triplet(sfr&, const char*) const: Assertion `parse_rc == 0' failed.

    Thread 1 "CSGOptiXRenderT" received signal SIGABRT, Aborted.
    0x00007ffff488bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64 libgcc-11.5.0-5.el9_5.alma.1.x86_64 libstdc++-11.5.0-5.el9_5.alma.1.x86_64 nvidia-driver-common-610.43.02-1.el9.x86_64 nvidia-driver-cuda-libs-610.43.02-1.el9.x86_64 nvidia-driver-libs-610.43.02-1.el9.x86_64 openssl-libs-3.5.1-7.el9_7.x86_64
    (gdb) bt
    #0  0x00007ffff488bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff483eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff4828833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff482875b in __assert_fail_base.cold () from /lib64/libc.so.6
    #4  0x00007ffff4837886 in __assert_fail () from /lib64/libc.so.6
    #5  0x000000000044f07b in stree::get_frame_from_triplet (this=0x5352b0, f=..., q_spec=0x42c9040 "sWorld:0:0") at /data1/blyth/local/opticks_Debug/include/SysRap/stree.h:2986
    #6  0x000000000044db49 in stree::get_frame (this=0x5352b0, q_spec_=0x42c9040 "sWorld:0:0") at /data1/blyth/local/opticks_Debug/include/SysRap/stree.h:2593
    #7  0x000000000045ae3d in SGLM::set_frame (this=0x558ff0, q_spec=0x42c9040 "sWorld:0:0") at /data1/blyth/local/opticks_Debug/include/SysRap/SGLM.h:1469
    #8  0x0000000000418645 in main (argc=1, argv=0x7fffffffbb38) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderTest.cc:208
    (gdb)




