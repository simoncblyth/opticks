Lifecycle_test_fails_from_SEventConfig__InputGenstepPathExists_with_null_path
================================================================================


Fixed::

    FAILS:  2   / 213   :  Thu Jan 25 10:33:29 2024   
      89 /107 Test #89 : SysRapTest.SEvt_Lifecycle_Test                Subprocess aborted***Exception:   0.10   
      11 /20  Test #11 : QUDARapTest.QEvent_Lifecycle_Test             ***Failed                      0.46   



::

    N[blyth@localhost tests]$ ./SEvt_Lifecycle_Test.sh dbg
                               arg : dbg 
              OPTICKS_INPUT_PHOTON : RainXZ100_f4.npy 
                OPTICKS_EVENT_MODE : DebugLite 
                               EVT : p001 
                               TMP : /home/blyth/tmp 
                              GEOM : SEVT_LIFECYCLE_TEST 
                           VERSION : 0 
                              FOLD : /home/blyth/tmp/GEOM/SEVT_LIFECYCLE_TEST/SEvt_Lifecycle_Test/ALL0/p001 
    gdb -ex r --args SEvt_Lifecycle_Test
    Thu Jan 25 11:08:12 CST 2024
    GNU gdb (GDB) 12.1
    Copyright (C) 2022 Free Software Foundation, Inc.
    ...

    SEvt::DescINSTANCE Count() 1
     Exists(0) YES
     Exists(1) NO 
    terminate called after throwing an instance of 'std::logic_error'
      what():  basic_string::_M_construct null not valid

    Program received signal SIGABRT, Aborted.
    0x00007ffff64fb387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff64fb387 in raise () from /lib64/libc.so.6
        dest=0x7ffff6c5afe0 <std::logic_error::~logic_error()>)
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_throw.cc:95
    #6  0x00007ffff6c3d121 in std::__throw_logic_error (__s=0x42de78 "basic_string::_M_construct null not valid")
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/src/c++11/functexcept.cc:70
    #7  0x0000000000419d87 in std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*> (this=0x7fffffff0e40, __beg=0x0, 
        __end=0x1 <error: Cannot access memory at address 0x1>) at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/include/c++/11.2.0/bits/basic_string.tcc:212
    #8  0x00007ffff7af1492 in std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string<std::allocator<char> > (this=0x7fffffff0e40, __s=0x0, 
        __a=...) at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/include/c++/11.2.0/bits/basic_string.h:539
    #9  0x00007ffff7b0cbfc in spath::_Join<char const*> () at /home/blyth/junotop/opticks/sysrap/spath.h:447
    #10 0x00007ffff7b0c657 in spath::_Resolve<char const*> () at /home/blyth/junotop/opticks/sysrap/spath.h:403
    #11 0x00007ffff7b0f54c in spath::Exists<char const*> () at /home/blyth/junotop/opticks/sysrap/spath.h:579
    #12 0x00007ffff7b9e895 in SEventConfig::InputGenstepPathExists (idx=0) at /home/blyth/junotop/opticks/sysrap/SEventConfig.cc:259
    #13 0x00007ffff7baa4dc in SEvt::hasInputGenstepPath (this=0x45d080) at /home/blyth/junotop/opticks/sysrap/SEvt.cc:428
    #14 0x00007ffff7bab2c9 in SEvt::addInputGenstep (this=0x45d080) at /home/blyth/junotop/opticks/sysrap/SEvt.cc:744
    #15 0x00007ffff7baea0a in SEvt::beginOfEvent (this=0x45d080, eventID=0) at /home/blyth/junotop/opticks/sysrap/SEvt.cc:1573
    #16 0x0000000000405cd4 in main (argc=1, argv=0x7fffffff25a8) at /home/blyth/junotop/opticks/sysrap/tests/SEvt_Lifecycle_Test.cc:24
    (gdb) f 15
    #15 0x00007ffff7baea0a in SEvt::beginOfEvent (this=0x45d080, eventID=0) at /home/blyth/junotop/opticks/sysrap/SEvt.cc:1573
    1573	    addInputGenstep();  // does genstep setup for simtrace, input photon and torch running
    (gdb) 





::

    [  2%] Generating OpticksPhoton_Abbrev.json
    [  2%] Generating OpticksGenstep_Enum.ini
    nvcc fatal   : Unsupported gpu architecture 'compute_'
    CMake Error at SysRap_generated_SU.cu.o.Debug.cmake:216 (message):
      Error generating
      /home/blyth/junotop/ExternalLibs/opticks/head/build/sysrap/CMakeFiles/SysRap.dir//./SysRap_generated_SU.cu.o


    make[2]: *** [CMakeFiles/SysRap.dir/SysRap_generated_SU.cu.o] Error 1
    make[1]: *** [CMakeFiles/SysRap.dir/all] Error 2
    make[1]: *** Waiting for unfinished jobs....
    [2024-01-25 11:23:29,102] p286384 {/home/blyth/junotop/opti



