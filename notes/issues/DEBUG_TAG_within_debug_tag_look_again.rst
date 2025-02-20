DEBUG_TAG_within_debug_tag_look_again
=======================================

Overview
-----------

* Again DEBUG_TAG usage of stagr within qsim::propagate_at_boundary resulting in pullback fails.
* Switching off DEBUG_TAG from sysrap/CMakeLists.txt avoids the issue
* Actual reason why DEBUG_TAG and stagr/tagr has started causing problems remains unknown 


Issue : three QSimTest_ALL.sh FAIL from TEST=propagate_at_boundary_s/p/x_polarized
-------------------------------------------------------------------------------------

These tests expand on some the above ctests particularly for 
executables that encompass many tests selected via TEST envvar 
which are simpler to run and debug outside of ctests::

    ~/o/qudarap/tests/QSimTest_ALL.sh    ## QSimTest tests


    Thu Feb 20 10:22:00 CST 2025
    TOTAL : 25 
    PASS  : 22 
    FAIL  : 3 
    === 013 === [ TEST=propagate_at_boundary_s_polarized /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
    === 013 === ] ***FAIL*** 
    === 014 === [ TEST=propagate_at_boundary_p_polarized /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
    === 014 === ] ***FAIL*** 
    === 015 === [ TEST=propagate_at_boundary_x_polarized /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
    === 015 === ] ***FAIL*** 


* see ~/o/notes/issues/QSimTest_ALL_initial_shakedown.rst  (2025/02/20 3/25 FAIL)

::

    Thread 1 "QSimTest" received signal SIGABRT, Aborted.
    0x00007ffff5bf6387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff5bf6387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff5bf7a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff633589a in __gnu_cxx::__verbose_terminate_handler () at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/vterminate.cc:95
    #3  0x00007ffff634136a in __cxxabiv1::__terminate (handler=<optimized out>) at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_terminate.cc:48
    #4  0x00007ffff63413d5 in std::terminate () at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_terminate.cc:58
    #5  0x00007ffff6341669 in __cxxabiv1::__cxa_throw (obj=<optimized out>, tinfo=0x7ffff7a4b338 <typeinfo for QUDA_Exception>, dest=0x7ffff75d4a44 <QUDA_Exception::~QUDA_Exception()>)
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_throw.cc:95
    #6  0x00007ffff75f34b9 in QU::copy_device_to_host_and_free<sphoton> (h=0x7fff982f6010, d=0x7fffa7e00000, num_items=1000000, label=0x7ffff76aef56 "QSim::photon_launch_mutate") at /home/blyth/opticks/qudarap/QU.cc:514
    #7  0x00007ffff7598e86 in QSim::photon_launch_mutate (this=0x118478d0, photon=0x7fff982f6010, num_photon=1000000, type=25) at /home/blyth/opticks/qudarap/QSim.cc:1153
    #8  0x000000000040d7b6 in QSimTest::photon_launch_mutate (this=0x7fffffff3f70) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:665
    #9  0x000000000040dc7f in QSimTest::main (this=0x7fffffff3f70) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:755
    #10 0x000000000040e21f in main (argc=1, argv=0x7fffffff4718) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:804
    (gdb) 



Skipping the launch avoids the problem::

    QSim__photon_launch_mutate_SKIP_LAUNCH=1 ~/o/qudarap/tests/propagate_at_boundary_s_polarized.sh dbg


Binary search reveals the problem is avoided when commenting the use of tagr in qsim::propagate_at_boundary 
which is surprising as that implies DEBUG_TAG not switched off ? 

* YEP : it was still enabled in sysrap/CMakeLists.txt
* DEBUG_TAG should only be left ON while doing random aligned running tests
* SWITCHING OFF DEBUG_TAG : GETS THIS TO PASS WITHOUT THE COMMENTING 



Sticky DEBUG_TAG setting ? 
------------------------------

::

    P[blyth@localhost opticks]$ opticks-f DEBUG_TAG
    ./CSGOptiX/CMakeLists.txt:# target_compile_definitions( ${name} PUBLIC DEBUG_TAG )     ## NOW FROM sysrap/CMakeLists.txt
    ./qudarap/QSim__Desc.sh:DEBUG_TAG
    ./qudarap/CMakeLists.txt:Global compile definitions such as DEBUG_TAG and DEBUG_PIDX are defined in sysrap/CMakeLists.txt 
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:    printf("//propagate_at_boundary.DEBUG_TAG ctx.idx %d base %p base.pidx %d \n", ctx.idx, base, base->pidx  ); 
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/qsim.h:#if !defined(PRODUCTION) && defined(DEBUG_TAG)
    ./qudarap/QSim.cc:#ifdef DEBUG_TAG
    ./qudarap/QSim.cc:       << "DEBUG_TAG"
    ./qudarap/QSim.cc:       << "NOT-DEBUG_TAG"
    ./qudarap/QSim.cu:uses tagr for DEBUG_TAG recording of random consumption.  

    ./sysrap/CMakeLists.txt:DEBUG_TAG 
    ./sysrap/CMakeLists.txt:      $<$<CONFIG:Debug>:DEBUG_TAG>


    ./sysrap/SEvt.cc:    ctx.end();   // copies {seq,sup} into evt->{seq,sup}[idx] (and tag, flat when DEBUG_TAG)
    ./sysrap/sctx.h:#ifdef DEBUG_TAG
    ./sysrap/sbuild.h:#if defined(DEBUG_TAG)
    ./sysrap/sbuild.h:    static constexpr const bool _DEBUG_TAG = true ; 
    ./sysrap/sbuild.h:    static constexpr const bool _DEBUG_TAG = false ; 
    ./sysrap/sbuild.h:       << " _DEBUG_TAG           : " << ( _DEBUG_TAG ? "YES" :  "NO " ) << "\n"
    ./sysrap/ssys__Desc.sh:DEBUG_TAG
    ./sysrap/tests/sbuild_test.sh:opt="-DCONFIG_Release -DDEBUG_TAG -DPRODUCTION -DRNG_XORWOW"
    ./sysrap/ssys.h:#ifdef DEBUG_TAG
    ./sysrap/ssys.h:       << "DEBUG_TAG"
    ./sysrap/ssys.h:       << "NOT:DEBUG_TAG"

    ./u4/CustomBoundary.h:#ifdef DEBUG_TAG
    ./u4/CustomBoundary.h:#ifdef DEBUG_TAG
    ./u4/CustomBoundary.h:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.cc:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.hh:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.hh:#ifdef DEBUG_TAG
    ./u4/InstrumentedG4OpBoundaryProcess.hh:#ifdef DEBUG_TAG 
    ./u4/Local_DsG4Scintillation.cc:#ifdef DEBUG_TAG
    ./u4/Local_DsG4Scintillation.cc:#ifdef DEBUG_TAG
    ./u4/Local_DsG4Scintillation.cc:#ifdef DEBUG_TAG
    ./u4/Local_DsG4Scintillation.cc:#ifdef DEBUG_TAG
    ./u4/Local_DsG4Scintillation.cc:#ifdef DEBUG_TAG
    ./u4/Local_DsG4Scintillation.cc:#ifdef DEBUG_TAG
    ./u4/Local_DsG4Scintillation.cc:#ifdef DEBUG_TAG
    ./u4/Local_DsG4Scintillation.cc:#ifdef DEBUG_TAG
    ./u4/ShimG4OpAbsorption.cc://#ifdef DEBUG_TAG
    ./u4/ShimG4OpRayleigh.cc:#ifdef DEBUG_TAG
    ./u4/ShimG4OpRayleigh.cc:#ifdef DEBUG_TAG
    ./u4/ShimG4OpRayleigh.cc:#ifdef DEBUG_TAG
    ./u4/ShimG4OpRayleigh.cc:#ifdef DEBUG_TAG
    ./u4/ShimG4OpRayleigh.cc:#ifdef DEBUG_TAG
    ./u4/ShimG4OpRayleigh.cc:#ifdef DEBUG_TAG
    ./u4/U4Physics.hh:#ifdef DEBUG_TAG
    ./u4/U4Physics.hh:#ifdef DEBUG_TAG
    ./u4/U4RandomDirection.hh:#ifdef DEBUG_TAG
    ./u4/U4RandomDirection.hh:#ifdef DEBUG_TAG
    ./u4/U4RandomDirection.hh:#ifdef DEBUG_TAG
    ./u4/U4RandomTools.hh:#ifdef DEBUG_TAG
    ./u4/U4RandomTools.hh:#ifdef DEBUG_TAG
    ./u4/U4Physics.cc:#ifdef DEBUG_TAG
    ./u4/U4Physics.cc:#if defined(DEBUG_TAG)
    ./u4/U4Physics.cc:    ss << "DEBUG_TAG" << std::endl ; 
    ./u4/U4Physics.cc:    ss << "NOT:DEBUG_TAG" << std::endl ; 
    ./u4/U4Physics.cc:#ifdef DEBUG_TAG
    ./u4/U4Physics.cc:#ifdef DEBUG_TAG
    ./u4/U4Random.hh:#ifdef DEBUG_TAG
    ./u4/U4Random.cc:#ifdef DEBUG_TAG
    ./u4/U4Random.cc:#ifdef DEBUG_TAG
    P[blyth@localhost opticks]$ 





