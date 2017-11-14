
osx-optix-411-no-binary-for-GPU
=================================


* reconfigured for OptiX 411 with *opticks-cmake-modify-ex3*

* get below ptx problem 

* however suspect that changing OptiX version is not something that  
  current cmake machinery (eg OptiX version detection) can handle 
  as a config modification : it being necessary to do a full opticks-configure
  with wiping of the build dir to manage the change correctly


TODO: 

  * try to find a faster way to jump OptiX versions, 
    perhaps via separate bdir/prefix for different OptiX vers ?



::

    op 

    ...

    2017-11-14 14:36:39.795 INFO  [4840800] [OContext::launch@294] OContext::launch COMPILE DONE
    2017-11-14 14:36:39.795 INFO  [4840800] [OContext::launch@296] OContext::launch COMPILE time: 5.2e-05
    2017-11-14 14:36:39.795 INFO  [4840800] [OContext::launch@305] OContext::launch PRELAUNCH START
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: ptxas application ptx input, line 1557; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1568; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1579; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 1590; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ...   
    ptxas application ptx input, line 4432; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
    ptxas application ptx input, line 4976; error   : Argument 1 of instruction 'tex': .texref or .u64 register expected
     returned (209): No binary for GPU)
    /Users/blyth/opticks/bin/op.sh: line 787: 26873 Abort trap: 6           /usr/local/opticks/lib/OKTest --rendermode +global,+in0,+in1,+in2,+in3,+in4,+axis
    /Users/blyth/opticks/bin/op.sh RC 134
    simon:sysrap blyth$ 
 





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




