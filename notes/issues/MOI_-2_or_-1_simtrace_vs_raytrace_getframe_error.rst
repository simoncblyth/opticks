MOI_-2_or_-1_simtrace_vs_raytrace_getframe_error
======================================================

Sometime need -1 and sometimes -2::

    #moi=sChimneyLS:0:-1     ## for simtrace yields identity transforms
    #moi=sChimneyLS:0:-2      ## for simtrace yields expected transforms 
    moi=sWaterTube:0:-2
    #moi=sWaterTube:0:-1


Inconsistency between simtrace and rendering regards the MOI to use. 
Rendering with "sWaterTube:0:-2" asserts from stree::get_frame::

    stree::get_frame_instanced FAIL missing_transform  lvid 139 lvid_ordinal 0 repeat_ordinal -2 w2m NO  m2w NO  ii -1
    stree::get_frame FAIL q_spec[sWaterTube:0:-2]
     THIS CAN BE CAUSED BY NOT USING REPEAT_ORDINAL -1 (LAST OF TRIPLET) FOR GLOBAL GEOMETRY 
    CSGOptiXRenderInteractiveTest: /data1/blyth/local/opticks_Debug/include/SysRap/stree.h:1864: sfr stree::get_frame(const char*) const: Assertion `get_rc == 0' failed.

    Program received signal SIGABRT, Aborted.
    0x00007ffff4c8b52c in __pthread_kill_implementation () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-125.el9_5.3.alma.1.x86_64 libX11-1.7.0-9.el9.x86_64 libXau-1.0.9-8.el9.x86_64 libXext-1.3.4-8.el9.x86_64 libgcc-11.5.0-5.el9_5.alma.1.x86_64 libglvnd-1.3.4-1.el9.x86_64 libglvnd-glx-1.3.4-1.el9.x86_64 libstdc++-11.5.0-5.el9_5.alma.1.x86_64 libxcb-1.13.1-9.el9.x86_64 openssl-libs-3.2.2-6.el9_5.1.x86_64
    (gdb) bt
    #0  0x00007ffff4c8b52c in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff4c3e686 in raise () from /lib64/libc.so.6
    #2  0x00007ffff4c28833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff4c2875b in __assert_fail_base.cold () from /lib64/libc.so.6
    #4  0x00007ffff4c373c6 in __assert_fail () from /lib64/libc.so.6
    #5  0x0000000000472615 in stree::get_frame (this=0x5202b0, q_spec=0x7fffffffbf5c "sWaterTube:0:-2") at /data1/blyth/local/opticks_Debug/include/SysRap/stree.h:1864
    #6  0x000000000047d5ca in SGLM::setTreeScene (this=0xf234250, _tree=0x5202b0, _scene=0x522f00) at /data1/blyth/local/opticks_Debug/include/SysRap/SGLM.h:829
    #7  0x0000000000486b89 in CSGOptiXRenderInteractiveTest::init (this=0x7fffffffb310) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:124
    #8  0x0000000000486a77 in CSGOptiXRenderInteractiveTest::CSGOptiXRenderInteractiveTest (this=0x7fffffffb310) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:113
    #9  0x000000000044479c in main (argc=1, argv=0x7fffffffb4d8) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:204
    (gdb) 





