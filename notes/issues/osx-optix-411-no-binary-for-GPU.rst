
osx-optix-411-no-binary-for-GPU
=================================


* reconfigured for OptiX 411

::

    opticks-cmake-modify-ex3(){

      local msg="=== $FUNCNAME : "
      local bdir=$(opticks-bdir)
      local bcache=$bdir/CMakeCache.txt
      [ ! -f "$bcache" ] && echo $msg requires a preexisting $bcache from prior opticks-cmake run && return 
      opticks-bcd

      echo $msg opticks-cmakecache-vars BEFORE MODIFY 
      opticks-cmakecache-vars 

      cmake \
           -DOptiX_INSTALL_DIR=/Developer/OptiX \
           -DCOMPUTE_CAPABILITY=30 \
              .   

      echo $msg opticks-cmakecache-vars AFTER MODIFY 
      opticks-cmakecache-vars 

    }



::

    op 

    ...

    2017-11-14 14:36:39.795 INFO  [4840800] [OContext::launch@284] OContext::launch VALIDATE time: 0.002153
    2017-11-14 14:36:39.795 INFO  [4840800] [OContext::launch@292] OContext::launch COMPILE START
    2017-11-14 14:36:39.795 INFO  [4840800] [OContext::launch@294] OContext::launch COMPILE DONE
    2017-11-14 14:36:39.795 INFO  [4840800] [OContext::launch@296] OContext::launch COMPILE time: 5.2e-05
    2017-11-14 14:36:39.795 INFO  [4840800] [OContext::launch@305] OContext::launch PRELAUNCH START
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: ptxas application ptx input, line 1557; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1568; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1579; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1590; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1634; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1647; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1660; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1673; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2113; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2115; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2117; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2119; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2151; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2153; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2155; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2157; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2597; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2608; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2619; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2630; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2672; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2685; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2698; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2711; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2897; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2910; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2923; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 2936; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 3248; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 3464; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 3477; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 3490; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 3503; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 4432; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 4976; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
     returned (209): No binary for GPU)
    /Users/blyth/opticks/bin/op.sh: line 787: 26873 Abort trap: 6           /usr/local/opticks/lib/OKTest --rendermode +global,+in0,+in1,+in2,+in3,+in4,+axis
    /Users/blyth/opticks/bin/op.sh RC 134
    simon:sysrap blyth$ 
    simon:sysrap blyth$ 

