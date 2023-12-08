SSim_serialize_assert
========================

::

    ./cxs_min.sh : run : delete prior LOGFILE CSGOptiXSMTest.log
    2023-12-08 10:15:44.841  841640628 : [./cxs_min.sh 
    2023-12-08 10:15:46.324 FATAL [440213] [SSim::serialize@388]  top non-null : cannot serialize twice 
    CSGOptiXSMTest: /home/blyth/junotop/opticks/sysrap/SSim.cc:389: void SSim::serialize(): Assertion `top == nullptr' failed.
    ./cxs_min.sh: line 254: 440213 Aborted                 (core dumped) $bin
    ./cxs_min.sh run error
    N[blyth@localhost opticks]$ 


::

    BP=SSim::serialize SSim=INFO ~/o/cxs_min.sh dbg 



Looks like double load of SSim ?::

    (gdb) bt
    #0  0x00007ffff71776c0 in SSim::init()@plt () from /data/blyth/junotop/ExternalLibs/opticks/head/lib/../lib64/libSysRap.so
    #1  0x00007ffff726a370 in SSim::SSim (this=0x4550c0) at /home/blyth/junotop/opticks/sysrap/SSim.cc:151
    #2  0x00007ffff726a164 in SSim::Load (base=0x7fffffffad99 "/home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0", reldir=0x7ffff7d4947f "CSGFoundry/SSim") at /home/blyth/junotop/opticks/sysrap/SSim.cc:126
    #3  0x00007ffff7c1e0a4 in CSGFoundry::Load_ () at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:2971
    #4  0x00007ffff7c1d441 in CSGFoundry::Load () at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:2857
    #5  0x00007ffff7e594dd in CSGOptiX::SimulateMain () at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:175
    #6  0x0000000000405b15 in main (argc=1, argv=0x7fffffff2268) at /home/blyth/junotop/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb) 



    (gdb) c
    Continuing.
    2023-12-08 10:29:14.688 INFO  [458139] [SSim::init@160] [ new scontext
    [New Thread 0x7fffecfbb700 (LWP 2505)]
    2023-12-08 10:29:14.805 INFO  [458139] [SSim::init@162] ] new scontext
    2023-12-08 10:29:14.805 INFO  [458139] [SSim::init@164] scontext::desc [1:NVIDIA_TITAN_RTX]
    all_devices
    [0:NVIDIA_TITAN_V 1:NVIDIA_TITAN_RTX]
    idx/ord/mpc/cc:0/0/80/70  11.784 GB  NVIDIA TITAN V
    idx/ord/mpc/cc:1/1/72/75  23.652 GB  NVIDIA TITAN RTX
    visible_devices
    [1:NVIDIA_TITAN_RTX]
    idx/ord/mpc/cc:0/1/72/75  23.652 GB  NVIDIA TITAN RTX

    2023-12-08 10:29:14.806 INFO  [458139] [SSim::load_@357] [
    2023-12-08 10:29:14.806 INFO  [458139] [SSim::load_@361] [ top.load [/home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0/CSGFoundry/SSim]
    2023-12-08 10:29:15.156 INFO  [458139] [SSim::load_@365] ] top.load [/home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0/CSGFoundry/SSim]
    2023-12-08 10:29:15.409 INFO  [458139] [SSim::load_@370] ]
    [Detaching after fork from child process 2509]

    Thread 1 "CSGOptiXSMTest" hit Breakpoint 1, 0x00007ffff71776c0 in SSim::init()@plt () from /data/blyth/junotop/ExternalLibs/opticks/head/lib/../lib64/libSysRap.so
    (gdb) bt
    #0  0x00007ffff71776c0 in SSim::init()@plt () from /data/blyth/junotop/ExternalLibs/opticks/head/lib/../lib64/libSysRap.so
    #1  0x00007ffff726a370 in SSim::SSim (this=0x6986530) at /home/blyth/junotop/opticks/sysrap/SSim.cc:151
    #2  0x00007ffff726a164 in SSim::Load (base=0x685f450 "/home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0/CSGFoundry", reldir=0x7ffff7d489d6 "SSim") at /home/blyth/junotop/opticks/sysrap/SSim.cc:126
    #3  0x00007ffff7c1caf0 in CSGFoundry::load (this=0x685f240, dir_=0x685f160 "/home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0/CSGFoundry") at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:2679
    #4  0x00007ffff7c1c187 in CSGFoundry::load (this=0x685f240, base_=0x7fffffffad99 "/home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0", rel=0x7ffff7d48834 "CSGFoundry")
        at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:2570
    #5  0x00007ffff7c1e321 in CSGFoundry::Load (base=0x7fffffffad99 "/home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0", rel=0x7ffff7d48834 "CSGFoundry") at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:2987
    #6  0x00007ffff7c1e269 in CSGFoundry::Load_ () at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:2977
    #7  0x00007ffff7c1d441 in CSGFoundry::Load () at /home/blyth/junotop/opticks/CSG/CSGFoundry.cc:2857
    #8  0x00007ffff7e594dd in CSGOptiX::SimulateMain () at /home/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:175
    #9  0x0000000000405b15 in main (argc=1, argv=0x7fffffff2268) at /home/blyth/junotop/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb) 

