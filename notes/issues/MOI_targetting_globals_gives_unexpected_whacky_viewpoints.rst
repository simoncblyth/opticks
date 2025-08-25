FIXED MOI_targetting_globals_gives_unexpected_whacky_viewpoints
=====================================================================


FIXED : problem due to incorrect missing default of 1.f in SGLM::EVec4 for the UP vector
-------------------------------------------------------------------------------------------

Found this with "gdb watch SGLM::UP.w" cxr_min.sh::

    334 if [ "${arg/gdb}" != "$arg" ]; then
    335    if [ -f "$LOG" ]; then
    336        echo $BASH_SOURCE : gdb : delete prior LOG $LOG
    337        rm "$LOG"
    338    fi
    339    gdb -ex "watch SGLM::UP.w" -ex r  $bin
    340    [ $? -ne 0 ] && echo $BASH_SOURCE gdb error && exit 1
    341 fi



::

         static glm::vec3 EVec3(const char* key, const char* fallback);
    -    static glm::vec4 EVec4(const char* key, const char* fallback, float missing=1.f );
    -    static glm::vec4 SVec4(const char* str, float missing=1.f );
    -    static glm::vec3 SVec3(const char* str, float missing=1.f );
    +    static glm::vec4 EVec4(const char* key, const char* fallback, float missing );
    +    static glm::vec4 SVec4(const char* str, float missing );
    +    static glm::vec3 SVec3(const char* str, float missing );


     glm::ivec2 SGLM::WH = EVec2i(kWH,"1920,1080") ;
    -glm::vec4  SGLM::CE = EVec4(kCE,"0,0,0,100") ;
    -glm::vec4  SGLM::EYE  = EVec4(kEYE, "-1,-1,0,1") ;
    -glm::vec4  SGLM::LOOK = EVec4(kLOOK, "0,0,0,1") ;
    -glm::vec4  SGLM::UP  =  EVec4(kUP,   "0,0,1,0") ;

    +
    +glm::vec4  SGLM::CE = EVec4(kCE,"0,0,0,100", 100.f) ;
    +
    +glm::vec4  SGLM::EYE  = EVec4(kEYE, "-1,-1,0,1", 1.f) ;
    +glm::vec4  SGLM::LOOK = EVec4(kLOOK, "0,0,0,1" , 1.f) ;
    +glm::vec4  SGLM::UP  =  EVec4(kUP,   "0,0,1,0" , 0.f) ;
    +




Note that SGLM::UP.w has become 1.f not 0.f ?
------------------------------------------------

::

    2025-08-22 15:13:25.250 INFO  [817610] [CSGFoundry::getFrameE@3656]  MOI PMT_20inch_pmt_solid_head:152:-2
    2025-08-22 15:13:25.250 INFO  [817610] [CSGFoundry::getFrame@3510] [CSGFoundry__getFrame_VERBOSE] YES frs PMT_20inch_pmt_solid_head:152:-2 looks_like_moi YES looks_like_raw NO 
    2025-08-22 15:13:25.250 INFO  [817610] [CSGFoundry::getFrame@3529] [CSGFoundry__getFrame_VERBOSE] YES frs PMT_20inch_pmt_solid_head:152:-2 looks_like_moi YES midx 48 mord 152 gord -2 rc 0
    2025-08-22 15:13:25.250 INFO  [817610] [CSGFoundry::getFrame@3539] [CSGFoundry__getFrame_VERBOSE] YES[fr.desc
    sframe::desc inst 0 frs -
     ekvid - ek - ev -
     ce  (-10780.880,18120.961,20163.000,309.640)  is_zero 0
     m2w ( 1.000, 0.000, 0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000, 1.000, 0.000) (-10780.880,18120.961,20163.000, 1.000) 
     w2m ( 1.000, 0.000, 0.000, 0.000) ( 0.000, 1.000, 0.000, 0.000) ( 0.000, 0.000, 1.000, 0.000) (10780.880,-18120.961,-20163.000, 1.000) 
     midx   48 mord  152 gord   -2
     inst    0
     ix0     0 ix1     0 iy0     0 iy1     0 iz0     0 iz1     0 num_photon    0
     ins     0 gas     0 sensor_identifier        0 sensor_index      0
     propagate_epsilon    0.00000 is_hostside_simtrace NO
    ]fr.desc

    CSGOptiXRenderInteractiveTest: /data1/blyth/local/opticks_Debug/include/SysRap/SGLM.h:1262: void SGLM::update(): Assertion `UP.w == 0.f' failed.

    Thread 1 "CSGOptiXRenderI" received signal SIGABRT, Aborted.
    0x00007ffff4a8b52c in __pthread_kill_implementation () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-125.el9_5.3.alma.1.x86_64 libX11-1.7.0-9.el9.x86_64 libXau-1.0.9-8.el9.x86_64 libXext-1.3.4-8.el9.x86_64 libgcc-11.5.0-5.el9_5.alma.1.x86_64 libglvnd-1.3.4-1.el9.x86_64 libglvnd-glx-1.3.4-1.el9.x86_64 libnvidia-ml-570.124.06-1.el9.x86_64 libstdc++-11.5.0-5.el9_5.alma.1.x86_64 libxcb-1.13.1-9.el9.x86_64 nvidia-driver-cuda-libs-570.124.06-1.el9.x86_64 nvidia-driver-libs-570.124.06-1.el9.x86_64 openssl-libs-3.2.2-6.el9_5.1.x86_64
    (gdb) bt
    #0  0x00007ffff4a8b52c in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff4a3e686 in raise () from /lib64/libc.so.6
    #2  0x00007ffff4a28833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff4a2875b in __assert_fail_base.cold () from /lib64/libc.so.6
    #4  0x00007ffff4a373c6 in __assert_fail () from /lib64/libc.so.6
    #5  0x000000000048777b in SGLM::update (this=0x1b072780) at /data1/blyth/local/opticks_Debug/include/SysRap/SGLM.h:1262
    #6  0x0000000000487595 in SGLM::set_frame (this=0x1b072780, fr_=...) at /data1/blyth/local/opticks_Debug/include/SysRap/SGLM.h:1183
    #7  0x00007ffff7e37057 in CSGOptiX::setFrame (this=0x1c232440, lfr=...) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:829
    #8  0x00007ffff7e36248 in CSGOptiX::initFrame (this=0x1c232440) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:630
    #9  0x00007ffff7e34a8f in CSGOptiX::init (this=0x1c232440) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:457
    #10 0x00007ffff7e345fa in CSGOptiX::CSGOptiX (this=0x1c232440, foundry_=0x14148150) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:428
    #11 0x00007ffff7e34076 in CSGOptiX::Create (fd=0x14148150) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:358
    #12 0x0000000000491630 in CSGOptiXRenderInteractiveTest::init (this=0x7fffffffb0e0) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:127
    #13 0x00000000004914ec in CSGOptiXRenderInteractiveTest::CSGOptiXRenderInteractiveTest (this=0x7fffffffb0e0) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:113
    #14 0x0000000000445e0e in main (argc=1, argv=0x7fffffffb288) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:206
    (gdb) 



watch SGLM::UP.w
---------------------

::

    (ok) A[blyth@localhost CSGOptiX]$ cxr_min.sh gdb
    vue_logging is a function
    vue_logging () 
    { 
        : ~/.opticks/GEOM/VUE.sh;
        type $FUNCNAME;
        export CSGFoundry__getFrame_VERBOSE=1;
        export CSGFoundry__getFrameE_VERBOSE=1;
        export SGLM_LEVEL=1
    }
    ...
    GNU gdb (AlmaLinux) 14.2-3.el9
    Reading symbols from CSGOptiXRenderInteractiveTest...

    (gdb) watch SGLM::UP.w
    Hardware watchpoint 1: SGLM::UP.w

    (gdb) r
    Starting program: /data1/blyth/local/opticks_Debug/lib/CSGOptiXRenderInteractiveTest 

    Hardware watchpoint 1: SGLM::UP.w

    Old value = 0
    New value = 1
    __static_initialization_and_destruction_0 (__initialize_p=1, __priority=65535) at /data1/blyth/local/opticks_Debug/include/SysRap/SGLM.h:676
    676	float      SGLM::ZOOM = EValue<float>(kZOOM, "1");
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-125.el9_5.3.alma.1.x86_64 libX11-1.7.0-9.el9.x86_64 libXau-1.0.9-8.el9.x86_64 libXext-1.3.4-8.el9.x86_64 libgcc-11.5.0-5.el9_5.alma.1.x86_64 libglvnd-1.3.4-1.el9.x86_64 libglvnd-glx-1.3.4-1.el9.x86_64 libstdc++-11.5.0-5.el9_5.alma.1.x86_64 libxcb-1.13.1-9.el9.x86_64 openssl-libs-3.2.2-6.el9_5.1.x86_64
    (gdb) bt
    #0  __static_initialization_and_destruction_0 (__initialize_p=1, __priority=65535) at /data1/blyth/local/opticks_Debug/include/SysRap/SGLM.h:676
    #1  0x00007ffff7e3bece in _GLOBAL__sub_I_CSGOptiX.cc(void) () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1439
    #2  0x00007ffff7fcc51e in call_init (env=0x7fffffffb298, argv=0x7fffffffb288, argc=1, l=<optimized out>) at dl-init.c:70
    #3  call_init (l=<optimized out>, argc=1, argv=0x7fffffffb288, env=0x7fffffffb298) at dl-init.c:26
    #4  0x00007ffff7fcc60c in _dl_init (main_map=0x7ffff7ffe210, argc=1, argv=0x7fffffffb288, env=0x7fffffffb298) at dl-init.c:117
    #5  0x00007ffff7fe488a in _dl_start_user () from /lib64/ld-linux-x86-64.so.2
    #6  0x0000000000000001 in ?? ()
    #7  0x00007fffffffb8b3 in ?? ()
    #8  0x0000000000000000 in ?? ()
    (gdb) 




