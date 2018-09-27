FIXED OKTest_blank_screen
============================


DONE
-------

* find a way to capture such problems into a test 

  * not very general, but now assert on invalid frames

FIXED BY
------------

1. setting up detector dependent default frame in OpticksResource which is applied when 
   the default torch config is used


Issue : genstep/resource generalizations for direct mode, have broken legacy mode 
------------------------------------------------------------------------------------

::

    OKTest 
        # blank : no geometry appears, after usual repeated Q 

    OKTest --tracer
        # geometry appears, O: raytrace triangulated works

    OKTest --compute --save
    OKTest --load --geocenter
        # geometry appears, no propagation on pressing A, no photon history 
        # issue with new gensteps approach ?



Part of issue is that changed the default frame for JUNO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

     22 // frame=3153,      DYB NEAR AD center
     23 // frame=62593      j1808 lAcrylic0x4bd3f20         ce  0.000   0.000   0.009 17820.008
     24 //    "frame=3153_"
     25 //    "frame=62593_"
     26 
     27 const char* TorchStepNPY::DEFAULT_CONFIG =
     28     "type=sphere_"
     29     "frame=3153_"
     30     "source=0,0,0_"
     31     "target=0,0,1_"
     32     "photons=10000_"


Added assert to make it easier to spot the issue of out-of-range frame volume index::

    494 glm::mat4 OpticksHub::getTransform(int index)
    495 {
    496     glm::mat4 vt ;
    497     if(index > -1)
    498     {
    499         GMergedMesh* mesh0 = getMergedMesh(0);
    500         float* transform = mesh0 ? mesh0->getTransform(index) : NULL ;
    501         assert( transform ) ;
    502 
    503         if(transform) vt = glm::make_mat4(transform) ;
    504     }
    505     return vt ;
    506 }


It gets tripped at OpticksGen instanciation::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff655dbb6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff657a6080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff655371ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff654ff1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001005f22e9 libOpticksGeo.dylib`OpticksHub::getTransform(this=0x000000010af5d5d0, index=62593) at OpticksHub.cc:501
        frame #5: 0x00000001005f754f libOpticksGeo.dylib`OpticksGen::targetGenstep(this=0x000000010af91690, gs=0x000000010af91710) at OpticksGen.cc:280
      * frame #6: 0x00000001005f6e14 libOpticksGeo.dylib`OpticksGen::makeTorchstep(this=0x000000010af91690) at OpticksGen.cc:342
        frame #7: 0x00000001005f6a04 libOpticksGeo.dylib`OpticksGen::makeLegacyGensteps(this=0x000000010af91690, code=4096) at OpticksGen.cc:185
        frame #8: 0x00000001005f6437 libOpticksGeo.dylib`OpticksGen::initFromLegacyGensteps(this=0x000000010af91690) at OpticksGen.cc:159
        frame #9: 0x00000001005f5c33 libOpticksGeo.dylib`OpticksGen::init(this=0x000000010af91690) at OpticksGen.cc:98
        frame #10: 0x00000001005f5b1d libOpticksGeo.dylib`OpticksGen::OpticksGen(this=0x000000010af91690, hub=0x000000010af5d5d0) at OpticksGen.cc:49
        frame #11: 0x00000001005f5c5d libOpticksGeo.dylib`OpticksGen::OpticksGen(this=0x000000010af91690, hub=0x000000010af5d5d0) at OpticksGen.cc:48
        frame #12: 0x00000001005ef2d8 libOpticksGeo.dylib`OpticksHub::init(this=0x000000010af5d5d0) at OpticksHub.cc:187
        frame #13: 0x00000001005ef01a libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x000000010af5d5d0, ok=0x000000010c800000) at OpticksHub.cc:156
        frame #14: 0x00000001005ef42d libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x000000010af5d5d0, ok=0x000000010c800000) at OpticksHub.cc:155
        frame #15: 0x00000001000d3d74 libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfe768, argc=1, argv=0x00007ffeefbfe820, argforced=0x0000000000000000) at OKMgr.cc:44
        frame #16: 0x00000001000d41bb libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfe768, argc=1, argv=0x00007ffeefbfe820, argforced=0x0000000000000000) at OKMgr.cc:52
        frame #17: 0x000000010000b94a OKTest`main(argc=1, argv=0x00007ffeefbfe820) at OKTest.cc:13
        frame #18: 0x00007fff6548b015 libdyld.dylib`start + 1
    (lldb) 




